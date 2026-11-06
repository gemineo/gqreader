// Command goldengen produces language-agnostic golden reference files from the
// Go implementation, so that ports to other languages can be validated stage by
// stage without needing this module (or OpenCV 4) at hand.
//
// For every annotated image of Dataset-QR-2020 it writes one JSON file with:
//   - the raw detector output (confidence, bbox, polygon, quad, padded quad),
//   - the end-to-end pipeline output (decoded content and winning strategy),
//   - the expected annotations and the pass/fail verdict.
//
// A summary.json file records the toolchain versions, configuration, model
// hash and aggregate accuracy.
//
// Usage:
//
//	go run ./cmd/goldengen -dataset testdata/Dataset-QR-2020 -out testdata/golden
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"

	gqreader "github.com/gemineo/gqreader"
	"github.com/gemineo/gqreader/qrdet"
)

// schemaVersion is bumped whenever the JSON layout changes incompatibly.
const schemaVersion = 1

type detectionJSON struct {
	Confidence   float32       `json:"confidence"`
	BBoxXYXY     [4]float32    `json:"bbox_xyxy"`
	QuadXY       [4][2]float32 `json:"quad_xy"`
	PaddedQuadXY [4][2]float32 `json:"padded_quad_xy"`
	PolygonXY    [][2]float32  `json:"polygon_xy"`
}

type resultJSON struct {
	Detection  detectionJSON `json:"detection"`
	Decoded    bool          `json:"decoded"`
	Content    string        `json:"content"`
	ContentHex string        `json:"content_hex"`
	ValidUTF8  bool          `json:"content_valid_utf8"`
	Strategy   string        `json:"strategy"`
	Matched    bool          `json:"matched_annotation"`
}

type imageJSON struct {
	SchemaVersion int             `json:"schema_version"`
	Image         string          `json:"image"`
	Width         int             `json:"width"`
	Height        int             `json:"height"`
	Expected      []string        `json:"expected"`
	Threshold     int             `json:"threshold"`
	MatchedCount  int             `json:"matched_count"`
	Passed        bool            `json:"passed"`
	Detections    []detectionJSON `json:"detections"`
	Results       []resultJSON    `json:"results"`
}

type summaryJSON struct {
	SchemaVersion int               `json:"schema_version"`
	GeneratedAt   string            `json:"generated_at"`
	GitCommit     string            `json:"git_commit"`
	Versions      map[string]string `json:"versions"`
	Model         map[string]string `json:"model"`
	Config        map[string]any    `json:"config"`
	Images        int               `json:"images"`
	ImagesPassed  int               `json:"images_passed"`
	QRExpected    int               `json:"qr_expected"`
	QRMatched     int               `json:"qr_matched"`
	Failed        []string          `json:"failed_images"`
	ThresholdRule string            `json:"threshold_rule"`
}

func main() {
	datasetDir := flag.String("dataset", "testdata/Dataset-QR-2020", "Dataset-QR-2020 directory")
	modelPath := flag.String("model", "qrdet/models/qrdet-s.onnx", "path to qrdet-s.onnx")
	outDir := flag.String("out", "testdata/golden", "output directory")
	flag.Parse()

	if err := run(*datasetDir, *modelPath, *outDir); err != nil {
		log.Fatal(err)
	}
}

func run(datasetDir, modelPath, outDir string) error {
	annotations, err := parseAnnotations(filepath.Join(datasetDir, "annotations.txt"))
	if err != nil {
		return err
	}

	// CPU provider: deterministic and representative of a portable port.
	detCfg := qrdet.DefaultConfig().WithProvider(qrdet.CPUProviderConfig())
	cfg := gqreader.DefaultConfig().WithDetectionConfig(detCfg)

	reader, err := gqreader.NewFromFile(modelPath, cfg)
	if err != nil {
		return fmt.Errorf("creating reader: %w", err)
	}
	defer reader.Close()

	detector, err := qrdet.NewDetectorFromFile(modelPath, detCfg)
	if err != nil {
		return fmt.Errorf("creating detector: %w", err)
	}
	defer detector.Close()
	defer func() { _ = qrdet.CleanupONNXRuntime() }()

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("creating output dir: %w", err)
	}

	stems := make([]string, 0, len(annotations))
	for stem := range annotations {
		stems = append(stems, stem)
	}
	sort.Strings(stems)

	sum := summaryJSON{
		SchemaVersion: schemaVersion,
		GeneratedAt:   time.Now().UTC().Format(time.RFC3339),
		GitCommit:     gitCommit(),
		ThresholdRule: "1-5 expected: >=1 matched; 6+ expected: >=ceil(30%)",
		Failed:        []string{},
	}

	for _, stem := range stems {
		imgPath := filepath.Join(datasetDir, stem+".jpg")
		if _, err := os.Stat(imgPath); err != nil {
			log.Printf("skipping %s: %v", stem, err)
			continue
		}
		rec, err := processImage(reader, detector, stem, imgPath, annotations[stem])
		if err != nil {
			return err
		}
		if err := writeJSON(filepath.Join(outDir, stem+".json"), rec); err != nil {
			return err
		}

		sum.Images++
		sum.QRExpected += len(rec.Expected)
		sum.QRMatched += rec.MatchedCount
		if rec.Passed {
			sum.ImagesPassed++
		} else {
			sum.Failed = append(sum.Failed, stem)
		}
		log.Printf("%-6s passed=%-5v matched=%d/%d detections=%d",
			stem, rec.Passed, rec.MatchedCount, len(rec.Expected), len(rec.Detections))
	}

	modelHash, err := fileSHA256(modelPath)
	if err != nil {
		return err
	}
	sum.Model = map[string]string{
		"file":   filepath.Base(modelPath),
		"sha256": modelHash,
		"source": "https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release/qrdet-s.pt (exported to ONNX)",
	}
	sum.Versions = map[string]string{
		"go":          runtime.Version(),
		"gocv":        gocv.Version(),
		"opencv":      gocv.OpenCVVersion(),
		"onnxruntime": ort.GetVersion(),
		"zbar":        pkgConfigVersion("zbar"),
	}
	sum.Config = map[string]any{
		"name":                 "DefaultConfig (ZBar primary, OpenCV fallback)",
		"provider":             "cpu",
		"confidence_threshold": detCfg.ConfidenceThreshold,
		"nms_iou_threshold":    detCfg.NMSIoUThreshold,
		"max_detections":       detCfg.MaxDetections,
		"input_size":           detCfg.InputSize,
		"scale_factors":        cfg.ScaleFactors,
		"try_perspective":      cfg.TryPerspective,
	}

	if err := writeJSON(filepath.Join(outDir, "summary.json"), sum); err != nil {
		return err
	}
	log.Printf("images passed %d/%d, QR codes matched %d/%d",
		sum.ImagesPassed, sum.Images, sum.QRMatched, sum.QRExpected)
	return nil
}

func processImage(reader *gqreader.QReader, detector *qrdet.Detector, stem, imgPath string, expected []string) (imageJSON, error) {
	dets, err := detector.DetectFile(imgPath)
	if err != nil {
		return imageJSON{}, fmt.Errorf("detecting %s: %w", stem, err)
	}
	results, err := reader.DetectAndDecodeFile(imgPath)
	if err != nil {
		return imageJSON{}, fmt.Errorf("decoding %s: %w", stem, err)
	}

	rec := imageJSON{
		SchemaVersion: schemaVersion,
		Image:         stem + ".jpg",
		Expected:      expected,
		Threshold:     successThreshold(len(expected)),
		Detections:    make([]detectionJSON, 0, len(dets)),
		Results:       make([]resultJSON, 0, len(results)),
	}
	if len(dets) > 0 {
		rec.Width, rec.Height = dets[0].ImageWidth, dets[0].ImageHeight
	} else {
		img := gocv.IMRead(imgPath, gocv.IMReadColor)
		rec.Width, rec.Height = img.Cols(), img.Rows()
		_ = img.Close()
	}

	for _, d := range dets {
		rec.Detections = append(rec.Detections, toDetectionJSON(d))
	}

	// Multiset matching: each expected entry can be consumed once.
	remaining := make(map[string]int, len(expected))
	for _, e := range expected {
		remaining[e]++
	}
	for _, r := range results {
		matched := false
		if r.Decoded && remaining[r.Content] > 0 {
			remaining[r.Content]--
			matched = true
			rec.MatchedCount++
		}
		rec.Results = append(rec.Results, resultJSON{
			Detection:  toDetectionJSON(r.Detection),
			Decoded:    r.Decoded,
			Content:    strings.ToValidUTF8(r.Content, "�"),
			ContentHex: hex.EncodeToString([]byte(r.Content)),
			ValidUTF8:  utf8.ValidString(r.Content),
			Strategy:   r.Strategy,
			Matched:    matched,
		})
	}
	rec.Passed = rec.MatchedCount >= rec.Threshold
	return rec, nil
}

func toDetectionJSON(d qrdet.Detection) detectionJSON {
	poly := d.PolygonXY
	if poly == nil {
		poly = [][2]float32{}
	}
	return detectionJSON{
		Confidence:   d.Confidence,
		BBoxXYXY:     d.BBoxXYXY,
		QuadXY:       d.QuadXY,
		PaddedQuadXY: d.PaddedQuadXY,
		PolygonXY:    poly,
	}
}

// parseAnnotations mirrors the parser in integration_test.go: one line per
// image, "<stem>=<content1>|<content2>|...", with \r and \n escapes.
func parseAnnotations(path string) (map[string][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening annotations: %w", err)
	}
	defer f.Close()

	unescape := strings.NewReplacer(`\r`, "\r", `\n`, "\n")
	result := make(map[string][]string)
	scanner := bufio.NewScanner(f)
	first := true
	for scanner.Scan() {
		line := scanner.Text()
		if first {
			line = strings.TrimPrefix(line, "\xef\xbb\xbf")
			first = false
		}
		line = strings.TrimSpace(line)
		stem, raw, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		parts := strings.Split(raw, "|")
		for i, p := range parts {
			parts[i] = unescape.Replace(p)
		}
		result[stem] = parts
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading annotations: %w", err)
	}
	return result, nil
}

// successThreshold mirrors integration_test.go.
func successThreshold(n int) int {
	if n <= 5 {
		return 1
	}
	return max((n*30+99)/100, 1)
}

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("encoding %s: %w", path, err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	return nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("opening model: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("hashing model: %w", err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func gitCommit() string {
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	commit := strings.TrimSpace(string(out))
	if status, err := exec.Command("git", "status", "--porcelain", "--untracked-files=no").Output(); err == nil && len(status) > 0 {
		commit += "-dirty"
	}
	return commit
}

func pkgConfigVersion(pkg string) string {
	out, err := exec.Command("pkg-config", "--modversion", pkg).Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}
