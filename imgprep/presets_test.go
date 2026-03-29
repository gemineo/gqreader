package imgprep

import (
	"context"
	"image"
	"image/color"
	"testing"

	"gocv.io/x/gocv"
)

func makePresetTestImage(width, height int) gocv.Mat {
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)
	img.SetTo(gocv.NewScalar(180, 130, 90, 0))
	gocv.Rectangle(&img,
		image.Rect(50, 50, 150, 100),
		color.RGBA{0, 0, 0, 0},
		-1,
	)
	return img
}

func TestAllPresetStrategies(t *testing.T) {
	img := makePresetTestImage(300, 300)
	defer img.Close()

	presets := []struct {
		name     string
		strategy Strategy
	}{
		{"fast", FastStrategy()},
		{"standard", StandardStrategy()},
		{"degraded", DegradedStrategy()},
		{"fax", FaxStrategy()},
		{"high_contrast", HighContrastStrategy()},
		{"sauvola", SauvolaStrategy()},
		{"wolf", WolfStrategy()},
		{"rotation", RotationStrategy()},
		{"deskew", DeskewStrategy()},
		{"blue_channel", BlueChannelStrategy()},
		{"nlmeans", NLMeansStrategy()},
	}

	for _, tt := range presets {
		t.Run(tt.name, func(t *testing.T) {
			if tt.strategy.Name() == "" {
				t.Error("Name() is empty")
			}

			result, err := tt.strategy.Apply(img)
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

func TestQRDecodeStrategies(t *testing.T) {
	strategies := QRDecodeStrategies()

	if len(strategies) != 4 {
		t.Fatalf("QRDecodeStrategies() returned %d strategies, want 4", len(strategies))
	}

	expectedNames := []string{"identity", "inversion", "gray_otsu_blur", "sharpen_gray_blur"}
	for i, s := range strategies {
		if s.Name() != expectedNames[i] {
			t.Errorf("Strategy[%d].Name() = %q, want %q", i, s.Name(), expectedNames[i])
		}
	}

	img := makePresetTestImage(200, 200)
	defer img.Close()

	for _, s := range strategies {
		t.Run(s.Name(), func(t *testing.T) {
			result, err := s.Apply(img)
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

// mockDet is a simple detection type for pipeline/aggregate tests.
type mockDet struct {
	id   int
	conf float32
}

func TestTopN(t *testing.T) {
	getConf := func(d mockDet) float32 { return d.conf }

	tests := []struct {
		name       string
		detections []mockDet
		n          int
		wantLen    int
		wantFirst  float32
	}{
		{
			"top 3 of 5",
			[]mockDet{{1, 0.5}, {2, 0.9}, {3, 0.7}, {4, 0.3}, {5, 0.8}},
			3, 3, 0.9,
		},
		{
			"top 5 of 2",
			[]mockDet{{1, 0.5}, {2, 0.9}},
			5, 2, 0.9,
		},
		{
			"top 3 of 0",
			[]mockDet{},
			3, 0, 0,
		},
		{
			"top 1",
			[]mockDet{{1, 0.5}, {2, 0.9}, {3, 0.7}},
			1, 1, 0.9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := TopN(tt.detections, tt.n, getConf)
			if len(result) != tt.wantLen {
				t.Fatalf("TopN length = %d, want %d", len(result), tt.wantLen)
			}
			if tt.wantLen > 0 && result[0].conf != tt.wantFirst {
				t.Errorf("TopN[0].conf = %v, want %v", result[0].conf, tt.wantFirst)
			}
			// Verify sorted descending
			for i := 1; i < len(result); i++ {
				if result[i].conf > result[i-1].conf {
					t.Errorf("TopN not sorted: [%d].conf=%v > [%d].conf=%v",
						i, result[i].conf, i-1, result[i-1].conf)
				}
			}
		})
	}
}

func TestPipeline_WithMaxWorkers(t *testing.T) {
	img := makePresetTestImage(200, 200)
	defer img.Close()

	pipeline := NewPipeline[mockDet](
		WithStrategies[mockDet](
			&Grayscale{},
			&GaussianBlur{KernelSize: 3},
			&OtsuThreshold{},
			&Inversion{},
		),
		WithMaxWorkers[mockDet](2),
	)

	detect := func(mat gocv.Mat) ([]mockDet, error) {
		return []mockDet{{id: 1, conf: 0.8}}, nil
	}

	results, err := pipeline.Process(context.Background(), img, detect)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	if len(results) != 4 {
		t.Errorf("Expected 4 results (one per strategy), got %d", len(results))
	}
}

func TestPipeline_EmptyImage(t *testing.T) {
	emptyImg := gocv.NewMat()
	defer emptyImg.Close()

	pipeline := NewPipeline[mockDet](
		WithStrategies[mockDet](&Grayscale{}),
	)

	detect := func(mat gocv.Mat) ([]mockDet, error) {
		return nil, nil
	}

	_, err := pipeline.Process(context.Background(), emptyImg, detect)
	if err == nil {
		t.Error("Process with empty image should return error")
	}
}

func TestPipeline_NoStrategies(t *testing.T) {
	img := makePresetTestImage(200, 200)
	defer img.Close()

	pipeline := NewPipeline[mockDet]()

	detect := func(mat gocv.Mat) ([]mockDet, error) {
		return nil, nil
	}

	_, err := pipeline.Process(context.Background(), img, detect)
	if err == nil {
		t.Error("Process with no strategies should return error")
	}
}

func TestPipeline_WithVariantGenerator(t *testing.T) {
	img := makePresetTestImage(200, 200)
	defer img.Close()

	pipeline := NewPipeline[mockDet](
		WithStrategies[mockDet](
			&RotationSearch{Angles: []float64{-5, 5}},
		),
	)

	detectCount := 0
	detect := func(mat gocv.Mat) ([]mockDet, error) {
		detectCount++
		return []mockDet{{id: detectCount, conf: 0.7}}, nil
	}

	results, err := pipeline.Process(context.Background(), img, detect)
	if err != nil {
		t.Fatalf("Process failed: %v", err)
	}

	if len(results) < 2 {
		t.Errorf("Expected at least 2 variant results, got %d", len(results))
	}

	for _, r := range results {
		if r.Variant == "" {
			t.Error("Expected non-empty variant name for RotationSearch results")
		}
	}
}

func TestStrategyResult_FullName(t *testing.T) {
	tests := []struct {
		name     string
		strategy string
		variant  string
		want     string
	}{
		{"no variant", "grayscale", "", "grayscale"},
		{"with variant", "rotation_search", "rot_5.0", "rotation_search/rot_5.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := StrategyResult[mockDet]{Strategy: tt.strategy, Variant: tt.variant}
			if got := r.FullName(); got != tt.want {
				t.Errorf("FullName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAggregatedResults_HasResults(t *testing.T) {
	empty := &AggregatedResults[mockDet]{AllDetections: []mockDet{}}
	if empty.HasResults() {
		t.Error("HasResults() on empty = true, want false")
	}

	withResults := &AggregatedResults[mockDet]{
		AllDetections: []mockDet{{id: 1, conf: 0.9}},
	}
	if !withResults.HasResults() {
		t.Error("HasResults() with detections = false, want true")
	}
}

func TestPipeline_Strategies(t *testing.T) {
	s1 := &Grayscale{}
	s2 := &Inversion{}

	pipeline := NewPipeline[mockDet](
		WithStrategies[mockDet](s1, s2),
	)

	strategies := pipeline.Strategies()
	if len(strategies) != 2 {
		t.Fatalf("Strategies() length = %d, want 2", len(strategies))
	}
	if strategies[0].Name() != "grayscale" {
		t.Errorf("Strategies[0].Name() = %q, want %q", strategies[0].Name(), "grayscale")
	}
	if strategies[1].Name() != "inversion" {
		t.Errorf("Strategies[1].Name() = %q, want %q", strategies[1].Name(), "inversion")
	}
}

func TestRotationStrategy_IsVariantGenerator(t *testing.T) {
	s := RotationStrategy()
	vg, ok := s.(VariantGenerator)
	if !ok {
		t.Fatal("RotationStrategy() should implement VariantGenerator")
	}

	img := makePresetTestImage(200, 200)
	defer img.Close()

	count := 0
	for _, mat := range vg.Variants(img) {
		if mat.Empty() {
			t.Error("Variant mat is empty")
		}
		mat.Close()
		count++
	}

	if count == 0 {
		t.Error("RotationStrategy should produce at least one variant")
	}
}

func TestAggregate_NoResults(t *testing.T) {
	agg := Aggregate[mockDet](nil)
	if agg.TotalDetections() != 0 {
		t.Errorf("TotalDetections() = %d, want 0", agg.TotalDetections())
	}
	if agg.BestStrategy() != "" {
		t.Errorf("BestStrategy() = %q, want empty", agg.BestStrategy())
	}
}

func TestDeduplicate_Empty(t *testing.T) {
	dedup := &DeduplicateFunc[mockDet]{
		KeyFn:   func(d mockDet) string { return "" },
		MergeFn: func(dups []mockDet) mockDet { return dups[0] },
	}

	result := Deduplicate[mockDet](nil, dedup)
	if result != nil {
		t.Errorf("Deduplicate(nil) = %v, want nil", result)
	}
}
