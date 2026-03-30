package imgprep

import (
	"iter"
	"strings"
	"testing"

	"gocv.io/x/gocv"
)

// makeStrategyTestImage creates a test image with some contrast patterns.
func makeStrategyTestImage(width, height int) gocv.Mat {
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)
	img.SetTo(gocv.NewScalar(200, 150, 100, 0))
	return img
}

// makeGrayTestImage creates a single-channel grayscale test image.
func makeGrayTestImage(width, height int) gocv.Mat {
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC1)
	img.SetTo(gocv.NewScalar(128, 0, 0, 0))
	return img
}

func TestOtsuThreshold(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &OtsuThreshold{}

	if s.Name() != "otsu" {
		t.Errorf("Name() = %q, want %q", s.Name(), "otsu")
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
	if result.Empty() {
		t.Error("Result is empty")
	}
}

func TestOtsuThreshold_Grayscale(t *testing.T) {
	img := makeGrayTestImage(200, 200)
	defer img.Close()

	s := &OtsuThreshold{}
	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
}

func TestFastNlMeansDenoising(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &FastNlMeansDenoising{H: 10}

	if !strings.Contains(s.Name(), "nlmeans") {
		t.Errorf("Name() = %q, should contain 'nlmeans'", s.Name())
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}
	// Converts to grayscale
	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
}

func TestFastNlMeansDenoising_DefaultH(t *testing.T) {
	s := &FastNlMeansDenoising{}
	if s.h() != 10 {
		t.Errorf("Default H = %v, want 10", s.h())
	}
}

func TestFastNlMeansDenoising_Grayscale(t *testing.T) {
	img := makeGrayTestImage(200, 200)
	defer img.Close()

	s := &FastNlMeansDenoising{H: 10}
	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
}

func TestMorphOpen(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &MorphOpen{KernelSize: 3}

	if !strings.Contains(s.Name(), "morph_open") {
		t.Errorf("Name() = %q, should contain 'morph_open'", s.Name())
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}
	if result.Rows() != 200 || result.Cols() != 200 {
		t.Errorf("Dimensions changed: got %dx%d", result.Cols(), result.Rows())
	}
}

func TestMorphOpen_DefaultKernel(t *testing.T) {
	s := &MorphOpen{}
	if s.kernelSize() != 3 {
		t.Errorf("Default KernelSize = %d, want 3", s.kernelSize())
	}
}

func TestMorphClose(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &MorphClose{KernelSize: 5}

	if !strings.Contains(s.Name(), "morph_close") {
		t.Errorf("Name() = %q, should contain 'morph_close'", s.Name())
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}
	if result.Rows() != 200 || result.Cols() != 200 {
		t.Errorf("Dimensions changed: got %dx%d", result.Cols(), result.Rows())
	}
}

func TestMorphClose_DefaultKernel(t *testing.T) {
	s := &MorphClose{}
	if s.kernelSize() != 3 {
		t.Errorf("Default KernelSize = %d, want 3", s.kernelSize())
	}
}

func TestBlueChannel(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &BlueChannel{}

	if s.Name() != "blue_channel" {
		t.Errorf("Name() = %q, want %q", s.Name(), "blue_channel")
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
}

func TestBlueChannel_Grayscale(t *testing.T) {
	img := makeGrayTestImage(200, 200)
	defer img.Close()

	s := &BlueChannel{}
	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
}

func TestInversion(t *testing.T) {
	img := gocv.NewMatWithSize(10, 10, gocv.MatTypeCV8UC1)
	defer img.Close()
	img.SetTo(gocv.NewScalar(200, 0, 0, 0))

	s := &Inversion{}

	if s.Name() != "inversion" {
		t.Errorf("Name() = %q, want %q", s.Name(), "inversion")
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	// 255 - 200 = 55
	val := result.GetUCharAt(0, 0)
	if val != 55 {
		t.Errorf("Inverted pixel = %d, want 55", val)
	}
}

func TestInversion_Color(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &Inversion{}
	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 3 {
		t.Errorf("Expected 3 channels, got %d", result.Channels())
	}
}

func TestSharpen(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &Sharpen{}

	if s.Name() != "sharpen" {
		t.Errorf("Name() = %q, want %q", s.Name(), "sharpen")
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Rows() != 200 || result.Cols() != 200 {
		t.Errorf("Dimensions changed: got %dx%d", result.Cols(), result.Rows())
	}
	if result.Channels() != img.Channels() {
		t.Errorf("Channels changed: got %d, want %d", result.Channels(), img.Channels())
	}
}

func TestSharpen_DefaultStrength(t *testing.T) {
	s := &Sharpen{}
	if s.strength() != 1.0 {
		t.Errorf("Default strength = %v, want 1.0", s.strength())
	}
}

func TestNiblackThreshold(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &NiblackThreshold{WindowSize: 15, K: -0.2}

	if !strings.Contains(s.Name(), "niblack") {
		t.Errorf("Name() = %q, should contain 'niblack'", s.Name())
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
	if result.Empty() {
		t.Error("Result is empty")
	}
}

func TestNiblackThreshold_Defaults(t *testing.T) {
	s := &NiblackThreshold{}
	if s.windowSize() != 15 {
		t.Errorf("Default windowSize = %d, want 15", s.windowSize())
	}
	if s.k() != -0.2 {
		t.Errorf("Default k = %v, want -0.2", s.k())
	}
}

func TestNiblackThreshold_EvenWindowSize(t *testing.T) {
	s := &NiblackThreshold{WindowSize: 14}
	if s.windowSize() != 15 {
		t.Errorf("Even windowSize 14 should become 15, got %d", s.windowSize())
	}
}

func TestWolfThreshold(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &WolfThreshold{WindowSize: 15, K: 0.5}

	if !strings.Contains(s.Name(), "wolf") {
		t.Errorf("Name() = %q, should contain 'wolf'", s.Name())
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Channels() != 1 {
		t.Errorf("Expected 1 channel, got %d", result.Channels())
	}
}

func TestWolfThreshold_Defaults(t *testing.T) {
	s := &WolfThreshold{}
	if s.windowSize() != 15 {
		t.Errorf("Default windowSize = %d, want 15", s.windowSize())
	}
	if s.k() != 0.5 {
		t.Errorf("Default k = %v, want 0.5", s.k())
	}
}

func TestScaleVariants(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &ScaleVariants{Factors: []float64{0.5, 2.0}}

	if s.Name() != "scale" {
		t.Errorf("Name() = %q, want %q", s.Name(), "scale")
	}

	// Test Apply returns first variant
	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Apply returned empty mat")
	}
	// First factor is 0.5 → 100x100
	if result.Cols() != 100 || result.Rows() != 100 {
		t.Errorf("Apply dimensions = %dx%d, want 100x100", result.Cols(), result.Rows())
	}
}

func TestScaleVariants_Variants(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &ScaleVariants{Factors: []float64{0.5, 2.0}}

	count := 0
	for name, mat := range s.Variants(img) {
		if name == "" {
			t.Error("Variant name is empty")
		}
		if mat.Empty() {
			t.Error("Variant mat is empty")
		}
		mat.Close()
		count++
	}

	if count != 2 {
		t.Errorf("Variants count = %d, want 2", count)
	}
}

func TestScaleVariants_Filtering(t *testing.T) {
	// Small image: 40x40. Factor 0.5 → 20x20 < MinSize 25 → skipped
	img := makeStrategyTestImage(40, 40)
	defer img.Close()

	s := &ScaleVariants{Factors: []float64{0.5, 2.0}}

	count := 0
	for _, mat := range s.Variants(img) {
		mat.Close()
		count++
	}

	if count != 1 {
		t.Errorf("Expected 1 valid variant (factor 2.0 only), got %d", count)
	}
}

func TestScaleVariants_Defaults(t *testing.T) {
	s := &ScaleVariants{}
	if len(s.factors()) != 5 {
		t.Errorf("Default factors length = %d, want 5", len(s.factors()))
	}
	if s.minSize() != 25 {
		t.Errorf("Default minSize = %d, want 25", s.minSize())
	}
	if s.maxSize() != 1024 {
		t.Errorf("Default maxSize = %d, want 1024", s.maxSize())
	}
}

func TestPerspectiveCorrection(t *testing.T) {
	img := makeStrategyTestImage(400, 400)
	defer img.Close()

	s := &PerspectiveCorrection{
		Quad: [4][2]float32{
			{50, 50}, {250, 50}, {250, 250}, {50, 250},
		},
	}

	if s.Name() != "perspective" {
		t.Errorf("Name() = %q, want %q", s.Name(), "perspective")
	}

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}

	// Output should be roughly 200x200 (the side length of the quad)
	if result.Cols() != 200 || result.Rows() != 200 {
		t.Errorf("Output dimensions = %dx%d, want 200x200", result.Cols(), result.Rows())
	}
}

func TestPerspectiveCorrection_ZeroQuad(t *testing.T) {
	img := makeStrategyTestImage(200, 200)
	defer img.Close()

	s := &PerspectiveCorrection{} // Zero quad → side = 0 → returns copy

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}
	// Should be a copy of input
	if result.Rows() != 200 || result.Cols() != 200 {
		t.Errorf("Expected copy dimensions 200x200, got %dx%d", result.Cols(), result.Rows())
	}
}

func TestStrategyFunc(t *testing.T) {
	s := NewStrategyFunc("my_func", func(src gocv.Mat) (gocv.Mat, error) {
		dst := gocv.NewMat()
		src.CopyTo(&dst)
		return dst, nil
	})

	if s.Name() != "my_func" {
		t.Errorf("Name() = %q, want %q", s.Name(), "my_func")
	}

	img := makeStrategyTestImage(100, 100)
	defer img.Close()

	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Result is empty")
	}
}

func TestNewVariantStrategy(t *testing.T) {
	gen := func(src gocv.Mat) iter.Seq2[string, gocv.Mat] {
		return func(yield func(string, gocv.Mat) bool) {
			for i, factor := range []float64{0.5, 1.5} {
				dst := gocv.NewMat()
				src.CopyTo(&dst)
				_ = factor
				if !yield("variant_"+string(rune('a'+i)), dst) {
					return
				}
			}
		}
	}

	s := NewVariantStrategy("test_variant", gen)

	if s.Name() != "test_variant" {
		t.Errorf("Name() = %q, want %q", s.Name(), "test_variant")
	}

	img := makeStrategyTestImage(100, 100)
	defer img.Close()

	// Test Apply returns first variant
	result, err := s.Apply(img)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	defer result.Close()

	if result.Empty() {
		t.Error("Apply returned empty mat")
	}

	// Test Variants iterator
	count := 0
	for _, mat := range s.Variants(img) {
		mat.Close()
		count++
	}
	if count != 2 {
		t.Errorf("Variants count = %d, want 2", count)
	}
}

// Test that all strategies return error on empty input.
func TestStrategies_EmptyImageError(t *testing.T) {
	emptyImg := gocv.NewMat()
	defer emptyImg.Close()

	strategies := []Strategy{
		&Grayscale{},
		&GaussianBlur{KernelSize: 3},
		&MedianBlur{KernelSize: 3},
		&FastNlMeansDenoising{H: 10},
		&CLAHE{ClipLimit: 2.0, TileSize: 8},
		&OtsuThreshold{},
		&AdaptiveThreshold{BlockSize: 11, C: 2},
		&MorphOpen{KernelSize: 3},
		&MorphClose{KernelSize: 3},
		&BlueChannel{},
		&Inversion{},
		&Sharpen{},
		&SauvolaThreshold{WindowSize: 15, K: 0.2},
		&NiblackThreshold{WindowSize: 15, K: -0.2},
		&WolfThreshold{WindowSize: 15, K: 0.5},
		&PerspectiveCorrection{Quad: [4][2]float32{{0, 0}, {100, 0}, {100, 100}, {0, 100}}},
		&Deskew{MaxAngle: 15},
	}

	for _, s := range strategies {
		t.Run(s.Name(), func(t *testing.T) {
			_, err := s.Apply(emptyImg)
			if err == nil {
				t.Errorf("%s.Apply(empty) should return error", s.Name())
			}
		})
	}
}

// Test kernel size normalization (even → odd).
func TestKernelSizeNormalization(t *testing.T) {
	tests := []struct {
		name     string
		strategy Strategy
		wantName string
	}{
		{"gaussian even=4", &GaussianBlur{KernelSize: 4}, "gaussian_5"},
		{"gaussian odd=5", &GaussianBlur{KernelSize: 5}, "gaussian_5"},
		{"gaussian default", &GaussianBlur{}, "gaussian_3"},
		{"median even=4", &MedianBlur{KernelSize: 4}, "median_5"},
		{"median default", &MedianBlur{}, "median_3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.strategy.Name() != tt.wantName {
				t.Errorf("Name() = %q, want %q", tt.strategy.Name(), tt.wantName)
			}
		})
	}
}

// Ensure ScaleVariants implements VariantGenerator at compile time.
var _ VariantGenerator = (*ScaleVariants)(nil)
