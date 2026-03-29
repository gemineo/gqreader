//go:build integration

package gqreader_test

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"

	gqreader "github.com/gemineo/gqreader"
	"github.com/gemineo/gqreader/qrdet"
)

var (
	sharedReader *gqreader.QReader
	readerOnce   sync.Once
	readerErr    error
)

// Dataset and model paths can be overridden via environment variables.
// This allows the test suite to run against an externally placed dataset
// (e.g. from the gqreader-benchmark companion project) without modifying
// the source.
//
//   QREADER_DATASET_DIR  path to the Dataset-QR-2020 directory
//                        (default: testdata/Dataset-QR-2020)
//   QREADER_MODEL_PATH   path to qrdet-s.onnx
//                        (default: qrdet/models/qrdet-s.onnx)
var (
	datasetDir = envOrDefault("QREADER_DATASET_DIR", "testdata/Dataset-QR-2020")
	modelPath  = envOrDefault("QREADER_MODEL_PATH", "qrdet/models/qrdet-s.onnx")
)

var annotationsFile = filepath.Join(datasetDir, "annotations.txt")

func envOrDefault(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func TestMain(m *testing.M) {
	readerOnce.Do(func() {
		// QREADER_PROVIDER selects the execution provider for the shared reader
		// used by tests and BenchmarkDataset. Other provider benchmarks create
		// their own readers.
		//
		// QREADER_DECODER selects the QR decoder:
		//   "zbar"   – libzbar (faster, same library as Python QReader)
		//   "opencv" – OpenCV QRCodeDetector (default, no extra deps)
		var cfg gqreader.Config
		if os.Getenv("QREADER_DECODER") == "zbar" {
			cfg = gqreader.ZBarConfig()
		} else {
			cfg = gqreader.DefaultConfig()
		}
		cfg.DetectionConfig = cfg.DetectionConfig.WithProvider(qrdet.ProviderFromEnv())
		sharedReader, readerErr = gqreader.NewFromFile(modelPath, cfg)
	})
	if readerErr != nil {
		fmt.Fprintf(os.Stderr, "SKIP: QReader init failed: %v\n", readerErr)
		os.Exit(0) // soft skip — do not fail CI when model is absent
	}
	code := m.Run()
	if sharedReader != nil {
		_ = sharedReader.Close()
	}
	_ = qrdet.CleanupONNXRuntime()
	os.Exit(code)
}

// parseAnnotations reads annotations.txt and returns a map from image stem
// (e.g., "QR2a") to the ordered list of expected QR code contents.
//
// File format: <stem>=<content1>|<content2>|...
//
// Multiple entries for the same stem represent multiple QR codes in one image.
// Entries with identical content are preserved (e.g., QR1e has 5 QR codes all
// encoding the same text; QR2b has 2 QR codes with the same URL). The count of
// entries is the total number of QR codes expected in the image.
//
// The first line may carry a UTF-8 BOM (0xEF 0xBB 0xBF) which is stripped.
func parseAnnotations(path string) (map[string][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening annotations: %w", err)
	}
	defer f.Close()

	result := make(map[string][]string)
	scanner := bufio.NewScanner(f)
	firstLine := true

	for scanner.Scan() {
		line := scanner.Text()

		// Strip UTF-8 BOM from the very first line.
		if firstLine {
			line = strings.TrimPrefix(line, "\xef\xbb\xbf")
			firstLine = false
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Split on the first '=' only — content may contain '=' characters.
		idx := strings.IndexByte(line, '=')
		if idx < 0 {
			continue
		}
		stem := line[:idx]
		rawContents := line[idx+1:]

		// Split on '|' without deduplication: multiple QR codes in one image
		// can legitimately encode the same content.
		parts := strings.Split(rawContents, "|")
		result[stem] = parts
	}

	return result, scanner.Err()
}

// successThreshold returns the minimum number of expected QR codes that must
// be decoded for a test case to pass.
//
//   - 1 expected  → require 1
//   - 2–5 expected → require at least 1
//   - 6+ expected  → require at least 30% (ceiling)
func successThreshold(n int) int {
	switch {
	case n <= 5:
		return 1
	default:
		// Integer ceiling of n * 0.30
		t := (n*30 + 99) / 100
		if t < 1 {
			return 1
		}
		return t
	}
}

// countMatched counts how many decoded results match entries in expected using
// multiset semantics. Each expected entry can be consumed at most once, so
// decoding 3 QRs all reading "foo" counts as 3 matches when expected contains
// "foo" three times.
func countMatched(results []gqreader.QRResult, expected []string) int {
	remaining := make(map[string]int, len(expected))
	for _, e := range expected {
		remaining[e]++
	}

	nMatched := 0
	for _, r := range results {
		if !r.Decoded {
			continue
		}
		if remaining[r.Content] > 0 {
			nMatched++
			remaining[r.Content]--
		}
	}
	return nMatched
}

// TestDatasetDecodeAll iterates over every annotated image and verifies that
// the decoded content meets the per-image success threshold.
//
// A summary of overall decode rates is logged after all sub-tests complete.
func TestDatasetDecodeAll(t *testing.T) {
	annotations, err := parseAnnotations(annotationsFile)
	if err != nil {
		t.Fatalf("parsing annotations: %v", err)
	}

	type testCase struct {
		stem    string
		imgPath string
		expected []string
	}

	cases := make([]testCase, 0, len(annotations))
	for stem, contents := range annotations {
		imgPath := filepath.Join(datasetDir, stem+".jpg")
		if _, err := os.Stat(imgPath); err != nil {
			t.Logf("skipping %s: image not found at %s", stem, imgPath)
			continue
		}
		cases = append(cases, testCase{stem: stem, imgPath: imgPath, expected: contents})
	}

	// Sort for deterministic sub-test ordering (map iteration is random).
	sort.Slice(cases, func(i, j int) bool { return cases[i].stem < cases[j].stem })

	var (
		totalImages   int
		imagesPassed  int
		totalExpected int
		totalDecoded  int
	)

	for _, tc := range cases {
		tc := tc
		t.Run(tc.stem, func(t *testing.T) {
			results, err := sharedReader.DetectAndDecodeFile(tc.imgPath)
			if err != nil {
				t.Errorf("%s: DetectAndDecodeFile error: %v", tc.stem, err)
				return
			}

			nExpected := len(tc.expected)
			threshold := successThreshold(nExpected)
			nMatched := countMatched(results, tc.expected)

			totalImages++
			totalExpected += nExpected
			totalDecoded += nMatched
			if nMatched >= threshold {
				imagesPassed++
			}

			if nMatched < threshold {
				t.Errorf("%s: decoded %d/%d expected QR codes (threshold %d); %d detections total",
					tc.stem, nMatched, nExpected, threshold, len(results))
			}
		})
	}

	decodeRate := 0.0
	if totalExpected > 0 {
		decodeRate = float64(totalDecoded) / float64(totalExpected) * 100
	}
	imgRate := 0.0
	if totalImages > 0 {
		imgRate = float64(imagesPassed) / float64(totalImages) * 100
	}
	t.Logf("=== Dataset Summary ===")
	t.Logf("Images tested:             %d", totalImages)
	t.Logf("Images meeting threshold:  %d/%d (%.1f%%)", imagesPassed, totalImages, imgRate)
	t.Logf("QR codes decoded:          %d/%d (%.1f%%)", totalDecoded, totalExpected, decodeRate)
}

// TestDatasetDetectionOnly verifies that the detector returns at least one
// result for every image in the dataset, exercising the ONNX inference and
// post-processing path independently of content correctness.
func TestDatasetDetectionOnly(t *testing.T) {
	entries, err := os.ReadDir(datasetDir)
	if err != nil {
		t.Fatalf("reading dataset dir: %v", err)
	}

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".jpg" {
			continue
		}
		name := entry.Name()
		stem := strings.TrimSuffix(name, ".jpg")
		imgPath := filepath.Join(datasetDir, name)

		t.Run(stem, func(t *testing.T) {
			results, err := sharedReader.DetectAndDecodeFile(imgPath)
			if err != nil {
				t.Errorf("%s: DetectAndDecodeFile error: %v", stem, err)
				return
			}
			if len(results) == 0 {
				t.Errorf("%s: expected at least one detection, got none", stem)
			}
		})
	}
}

// BenchmarkDataset measures end-to-end throughput of DetectAndDecodeFile over
// the full dataset using DefaultConfig.
//
// Run with:
//
//	go test -tags integration -bench=BenchmarkDataset -benchmem -benchtime=3x .
//
// The benchmark reports ns/op for one complete pass over the dataset.
// Per-image latency = ns/op / images/op.
func BenchmarkDataset(b *testing.B) {
	if sharedReader == nil {
		b.Skip("QReader not available")
	}

	imagePaths := collectImagePaths(b)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, path := range imagePaths {
			if _, err := sharedReader.DetectAndDecodeFile(path); err != nil {
				b.Errorf("DetectAndDecodeFile(%s): %v", path, err)
			}
		}
	}
	b.ReportMetric(float64(len(imagePaths)), "images/op")
}

// BenchmarkDatasetMinimal benchmarks the fast path using MinimalConfig
// (no scaling, no perspective correction). Compare against BenchmarkDataset
// to measure the cost of the full preprocessing pipeline.
func BenchmarkDatasetMinimal(b *testing.B) {
	imagePaths := collectImagePaths(b)

	minReader, err := gqreader.NewFromFile(modelPath, gqreader.MinimalConfig())
	if err != nil {
		b.Skipf("could not create minimal QReader: %v", err)
	}
	defer minReader.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, path := range imagePaths {
			if _, err := minReader.DetectAndDecodeFile(path); err != nil {
				b.Errorf("DetectAndDecodeFile(%s): %v", path, err)
			}
		}
	}
	b.ReportMetric(float64(len(imagePaths)), "images/op")
}

// BenchmarkDatasetZBar measures end-to-end throughput using the ZBar decoder
// (libzbar) instead of OpenCV QRCodeDetector.
// ZBar is the same decoder used by Python QReader (pyzbar) and is typically
// 3-5× faster per decode attempt.
//
// Run with:
//
//	go test -tags integration -bench=BenchmarkDatasetZBar -benchmem -benchtime=3x .
func BenchmarkDatasetZBar(b *testing.B) {
	imagePaths := collectImagePaths(b)

	r, err := gqreader.NewFromFile(modelPath, gqreader.ZBarConfig())
	if err != nil {
		b.Skipf("ZBar decoder unavailable: %v", err)
	}
	defer r.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, path := range imagePaths {
			if _, err := r.DetectAndDecodeFile(path); err != nil {
				b.Errorf("DetectAndDecodeFile(%s): %v", path, err)
			}
		}
	}
	b.ReportMetric(float64(len(imagePaths)), "images/op")
}

// BenchmarkDatasetZBarCoreML combines ZBar decoder with CoreML inference provider.
// This stacks both optimizations: fast inference (CoreML) + fast decode (ZBar).
func BenchmarkDatasetZBarCoreML(b *testing.B) {
	imagePaths := collectImagePaths(b)

	cfg := gqreader.ZBarConfig()
	cfg.DetectionConfig = cfg.DetectionConfig.WithProvider(qrdet.CoreMLProviderConfig())
	r, err := gqreader.NewFromFile(modelPath, cfg)
	if err != nil {
		b.Skipf("ZBar+CoreML unavailable: %v", err)
	}
	defer r.Close()

	// Warm up CoreML JIT compilation.
	if len(imagePaths) > 0 {
		_, _ = r.DetectAndDecodeFile(imagePaths[0])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, path := range imagePaths {
			if _, err := r.DetectAndDecodeFile(path); err != nil {
				b.Errorf("DetectAndDecodeFile(%s): %v", path, err)
			}
		}
	}
	b.ReportMetric(float64(len(imagePaths)), "images/op")
}

// BenchmarkDatasetCoreML measures end-to-end throughput using the CoreML execution
// provider with all compute units (CPU + GPU + Neural Engine).
// Compare against BenchmarkDataset to quantify the hardware acceleration gain.
//
// Run with:
//
//	go test -tags integration -bench=BenchmarkDataset -benchmem -benchtime=3x .
func BenchmarkDatasetCoreML(b *testing.B) {
	imagePaths := collectImagePaths(b)

	cfg := gqreader.DefaultConfig()
	cfg.DetectionConfig = cfg.DetectionConfig.WithProvider(qrdet.CoreMLProviderConfig())
	r, err := gqreader.NewFromFile(modelPath, cfg)
	if err != nil {
		b.Skipf("CoreML provider unavailable: %v", err)
	}
	defer r.Close()

	// Warm up: first inference triggers CoreML JIT compilation.
	if len(imagePaths) > 0 {
		_, _ = r.DetectAndDecodeFile(imagePaths[0])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, path := range imagePaths {
			if _, err := r.DetectAndDecodeFile(path); err != nil {
				b.Errorf("DetectAndDecodeFile(%s): %v", path, err)
			}
		}
	}
	b.ReportMetric(float64(len(imagePaths)), "images/op")
}

// BenchmarkDatasetCoreMLCPU measures throughput using CoreML with CPU-only compute units.
// This isolates CoreML graph compilation overhead from hardware acceleration: if this is
// close to BenchmarkDataset (plain CPU), the speedup in CoreML ALL comes from GPU/ANE.
func BenchmarkDatasetCoreMLCPU(b *testing.B) {
	imagePaths := collectImagePaths(b)

	cfg := gqreader.DefaultConfig()
	cfg.DetectionConfig = cfg.DetectionConfig.WithProvider(qrdet.CoreMLCPUOnlyConfig())
	r, err := gqreader.NewFromFile(modelPath, cfg)
	if err != nil {
		b.Skipf("CoreML+CPU provider unavailable: %v", err)
	}
	defer r.Close()

	if len(imagePaths) > 0 {
		_, _ = r.DetectAndDecodeFile(imagePaths[0])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, path := range imagePaths {
			if _, err := r.DetectAndDecodeFile(path); err != nil {
				b.Errorf("DetectAndDecodeFile(%s): %v", path, err)
			}
		}
	}
	b.ReportMetric(float64(len(imagePaths)), "images/op")
}

// BenchmarkDatasetCoreMLGPU measures throughput using CoreML with CPU and GPU.
// Useful for comparing ANE vs GPU dispatch on Apple Silicon.
func BenchmarkDatasetCoreMLGPU(b *testing.B) {
	imagePaths := collectImagePaths(b)

	cfg := gqreader.DefaultConfig()
	cfg.DetectionConfig = cfg.DetectionConfig.WithProvider(qrdet.CoreMLGPUConfig())
	r, err := gqreader.NewFromFile(modelPath, cfg)
	if err != nil {
		b.Skipf("CoreML+GPU provider unavailable: %v", err)
	}
	defer r.Close()

	if len(imagePaths) > 0 {
		_, _ = r.DetectAndDecodeFile(imagePaths[0])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, path := range imagePaths {
			if _, err := r.DetectAndDecodeFile(path); err != nil {
				b.Errorf("DetectAndDecodeFile(%s): %v", path, err)
			}
		}
	}
	b.ReportMetric(float64(len(imagePaths)), "images/op")
}

// collectImagePaths returns the sorted list of .jpg paths in datasetDir.
func collectImagePaths(tb testing.TB) []string {
	tb.Helper()
	entries, err := os.ReadDir(datasetDir)
	if err != nil {
		tb.Fatalf("reading dataset dir: %v", err)
	}
	var paths []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".jpg" {
			paths = append(paths, filepath.Join(datasetDir, e.Name()))
		}
	}
	return paths
}
