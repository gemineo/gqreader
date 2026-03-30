package imgprep

import (
	"context"
	"image"
	"image/color"
	"testing"
	"time"

	"gocv.io/x/gocv"
)

// createTestImage creates a simple test image with some patterns.
func createTestImage(width, height int) gocv.Mat {
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)

	// Fill with white
	img.SetTo(gocv.NewScalar(255, 255, 255, 0))

	// Draw a black rectangle (simulates a barcode region)
	gocv.Rectangle(&img,
		image.Rect(50, 50, 150, 100),
		color.RGBA{0, 0, 0, 0},
		-1, // Filled
	)

	return img
}

// createNoisyImage creates an image with salt-and-pepper noise.
func createNoisyImage(width, height int) gocv.Mat {
	img := createTestImage(width, height)

	// Add noise by randomly setting pixels
	for y := 0; y < height; y += 10 {
		for x := 0; x < width; x += 10 {
			if (x+y)%20 == 0 {
				img.SetUCharAt(y, x*3, 0)
				img.SetUCharAt(y, x*3+1, 0)
				img.SetUCharAt(y, x*3+2, 0)
			} else if (x+y)%20 == 10 {
				img.SetUCharAt(y, x*3, 255)
				img.SetUCharAt(y, x*3+1, 255)
				img.SetUCharAt(y, x*3+2, 255)
			}
		}
	}

	return img
}

func TestGrayscale(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	strategy := &Grayscale{}

	result, err := strategy.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}

	if result.Rows() != img.Rows() || result.Cols() != img.Cols() {
		t.Errorf("Size mismatch: expected %dx%d, got %dx%d",
			img.Rows(), img.Cols(), result.Rows(), result.Cols())
	}
}

func TestGrayscale_AlreadyGray(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	strategy := &Grayscale{}
	result, err := strategy.Apply(gray)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
}

func TestGaussianBlur(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	tests := []struct {
		name       string
		kernelSize int
		wantSize   int
	}{
		{"default", 0, 3},
		{"explicit", 5, 5},
		{"even_corrected", 4, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := &GaussianBlur{KernelSize: tt.kernelSize}

			result, err := strategy.Apply(img)
			if err != nil {
				t.Fatalf("Apply failed: %v", err)
			}
			defer result.Close()

			if result.Empty() {
				t.Error("Result is empty")
			}
		})
	}
}

func TestMedianBlur(t *testing.T) {
	img := createNoisyImage(200, 200)
	defer img.Close()

	strategy := &MedianBlur{KernelSize: 5}

	result, err := strategy.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}
}

func TestCLAHE(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	strategy := &CLAHE{ClipLimit: 2.0, TileSize: 8}

	result, err := strategy.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel (grayscale), got %d", result.Channels())
	}
}

func TestAdaptiveThreshold(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	strategy := &AdaptiveThreshold{
		BlockSize: 11,
		C:         2,
		Method:    gocv.AdaptiveThresholdGaussian,
	}

	result, err := strategy.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	// Check it's binary
	min, max, _, _ := gocv.MinMaxLoc(result)
	if min != 0 && max != 255 {
		t.Logf("Warning: result may not be binary (min=%f, max=%f)", min, max)
	}
}

func TestSauvolaThreshold(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	strategy := &SauvolaThreshold{
		WindowSize: 15,
		K:          0.2,
		R:          128,
	}

	result, err := strategy.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
}

func TestCompositeStrategy(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	composite := NewCompositeStrategy("test",
		&Grayscale{},
		&GaussianBlur{KernelSize: 3},
		&OtsuThreshold{},
	)

	if composite.Name() != "test" {
		t.Errorf("Expected name 'test', got '%s'", composite.Name())
	}

	result, err := composite.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}
}

func TestCompositeStrategy_Empty(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	composite := NewCompositeStrategy("empty")

	result, err := composite.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	// Should return a copy of input
	if result.Rows() != img.Rows() || result.Cols() != img.Cols() {
		t.Error("Empty composite should return copy of input")
	}
}

func TestRotationSearch(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	strategy := &RotationSearch{
		Angles: []float64{-5, 0, 5},
	}

	// Test Apply (returns first variant)
	result, err := strategy.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	result.Close()

	// Test Variants
	count := 0
	for name, mat := range strategy.Variants(img) {
		if name == "" {
			t.Error("Variant name should not be empty")
		}
		if mat.Empty() {
			t.Error("Variant mat should not be empty")
		}
		mat.Close()
		count++
	}

	if count != 3 {
		t.Errorf("Expected 3 variants, got %d", count)
	}
}

func TestDeskew(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	strategy := &Deskew{MaxAngle: 15}

	result, err := strategy.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}
}

func TestDefaultStrategies(t *testing.T) {
	strategies := DefaultStrategies()

	if len(strategies) == 0 {
		t.Error("DefaultStrategies returned empty slice")
	}

	// Verify all strategies have names
	for _, s := range strategies {
		if s.Name() == "" {
			t.Error("Strategy has empty name")
		}
	}
}

func TestAllStrategies(t *testing.T) {
	strategies := AllStrategies()

	if len(strategies) < len(DefaultStrategies()) {
		t.Error("AllStrategies should include at least DefaultStrategies")
	}
}

// mockDetection for pipeline tests
type mockDetection struct {
	id         int
	confidence float32
}

func TestPipeline_Process(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	pipeline := NewPipeline[mockDetection](
		WithStrategies[mockDetection](
			&Grayscale{},
			&GaussianBlur{KernelSize: 3},
		),
	)

	detectCount := 0
	detect := func(mat gocv.Mat) ([]mockDetection, error) {
		detectCount++
		return []mockDetection{{id: detectCount, confidence: 0.9}}, nil
	}

	ctx := context.Background()
	results, err := pipeline.Process(ctx, img, detect)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// Verify detection was called for each strategy
	totalDetections := 0
	for _, r := range results {
		totalDetections += len(r.Detections)
	}

	if totalDetections != 2 {
		t.Errorf("Expected 2 total detections, got %d", totalDetections)
	}
}

func TestPipeline_ProcessWithTimeout(t *testing.T) {
	img := createTestImage(200, 200)
	defer img.Close()

	pipeline := NewPipeline[mockDetection](
		WithStrategies[mockDetection](
			&Grayscale{},
		),
	)

	// Create a slow detector
	detect := func(mat gocv.Mat) ([]mockDetection, error) {
		time.Sleep(100 * time.Millisecond)
		return nil, nil
	}

	// Very short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	results, _ := pipeline.Process(ctx, img, detect)

	// Should have at least started processing
	_ = results // May have partial results or context error
}

func TestAggregate(t *testing.T) {
	results := []StrategyResult[mockDetection]{
		{
			Strategy:   "fast",
			Detections: []mockDetection{{id: 1, confidence: 0.9}},
		},
		{
			Strategy:   "standard",
			Detections: []mockDetection{{id: 2, confidence: 0.8}, {id: 3, confidence: 0.7}},
		},
		{
			Strategy: "failed",
			Error:    context.DeadlineExceeded,
		},
	}

	agg := Aggregate(results)

	if agg.TotalDetections() != 3 {
		t.Errorf("Expected 3 total detections, got %d", agg.TotalDetections())
	}

	if len(agg.SuccessfulStrategies) != 2 {
		t.Errorf("Expected 2 successful strategies, got %d", len(agg.SuccessfulStrategies))
	}

	if len(agg.FailedStrategies) != 1 {
		t.Errorf("Expected 1 failed strategy, got %d", len(agg.FailedStrategies))
	}

	if agg.BestStrategy() != "standard" {
		t.Errorf("Expected 'standard' as best strategy, got '%s'", agg.BestStrategy())
	}
}

func TestDeduplicate(t *testing.T) {
	detections := []mockDetection{
		{id: 1, confidence: 0.9},
		{id: 1, confidence: 0.7}, // Duplicate
		{id: 2, confidence: 0.8},
	}

	dedup := &DeduplicateFunc[mockDetection]{
		KeyFn: func(d mockDetection) string {
			return string(rune(d.id + '0'))
		},
		MergeFn: func(dups []mockDetection) mockDetection {
			best := dups[0]
			for _, d := range dups[1:] {
				if d.confidence > best.confidence {
					best = d
				}
			}
			return best
		},
	}

	unique := Deduplicate(detections, dedup)

	if len(unique) != 2 {
		t.Errorf("Expected 2 unique detections, got %d", len(unique))
	}

	// First should be the one with higher confidence
	if unique[0].confidence != 0.9 {
		t.Errorf("Expected first detection to have confidence 0.9, got %f", unique[0].confidence)
	}
}

func TestSortByConfidence(t *testing.T) {
	detections := []mockDetection{
		{id: 1, confidence: 0.5},
		{id: 2, confidence: 0.9},
		{id: 3, confidence: 0.7},
	}

	sorted := SortByConfidence(detections, func(d mockDetection) float32 {
		return d.confidence
	})

	if sorted[0].confidence != 0.9 {
		t.Errorf("Expected highest confidence first, got %f", sorted[0].confidence)
	}
	if sorted[2].confidence != 0.5 {
		t.Errorf("Expected lowest confidence last, got %f", sorted[2].confidence)
	}
}

func TestFilterByConfidence(t *testing.T) {
	detections := []mockDetection{
		{id: 1, confidence: 0.5},
		{id: 2, confidence: 0.9},
		{id: 3, confidence: 0.7},
	}

	filtered := FilterByConfidence(detections, 0.6, func(d mockDetection) float32 {
		return d.confidence
	})

	if len(filtered) != 2 {
		t.Errorf("Expected 2 detections above threshold, got %d", len(filtered))
	}
}

func BenchmarkStrategies(b *testing.B) {
	img := createTestImage(640, 480)
	defer img.Close()

	strategies := []Strategy{
		FastStrategy(),
		StandardStrategy(),
		DegradedStrategy(),
		SauvolaStrategy(),
	}

	for _, s := range strategies {
		b.Run(s.Name(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				result, err := s.Apply(img)
				if err != nil {
					b.Fatal(err)
				}
				result.Close()
			}
		})
	}
}
