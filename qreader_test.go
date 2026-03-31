package gqreader

import (
	"fmt"
	"math"
	"testing"

	"gocv.io/x/gocv"

	"github.com/gemineo/gqreader/qrdecode"
	"github.com/gemineo/gqreader/qrdet"
)

// mockDecoder implements qrdecode.Decoder for testing.
type mockDecoder struct {
	results []qrdecode.DecodedQR
	err     error
}

func (m *mockDecoder) Decode(img gocv.Mat) ([]qrdecode.DecodedQR, error) {
	return m.results, m.err
}

func (m *mockDecoder) Close() error { return nil }

func TestClampInt(t *testing.T) {
	tests := []struct {
		v, lo, hi int
		want      int
	}{
		{5, 0, 10, 5},
		{-1, 0, 10, 0},
		{10, 0, 10, 10}, // v >= hi → hi
		{15, 0, 10, 10},
		{0, 0, 10, 0},
		{9, 0, 10, 9},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("clamp(%d,%d,%d)", tt.v, tt.lo, tt.hi)
		t.Run(name, func(t *testing.T) {
			if got := clampInt(tt.v, tt.lo, tt.hi); got != tt.want {
				t.Errorf("clampInt(%d, %d, %d) = %d, want %d", tt.v, tt.lo, tt.hi, got, tt.want)
			}
		})
	}
}

func TestHypot(t *testing.T) {
	tests := []struct {
		dx, dy float32
		want   float32
	}{
		{3, 4, 5},
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
		{5, 12, 13},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("hypot(%v,%v)", tt.dx, tt.dy)
		t.Run(name, func(t *testing.T) {
			got := hypot(tt.dx, tt.dy)
			if math.Abs(float64(got-tt.want)) > 1e-3 {
				t.Errorf("hypot(%v, %v) = %v, want %v", tt.dx, tt.dy, got, tt.want)
			}
		})
	}
}

func TestPerspectiveSide(t *testing.T) {
	tests := []struct {
		name string
		quad [4][2]float32
		want int
	}{
		{
			"square 100x100",
			[4][2]float32{{0, 0}, {100, 0}, {100, 100}, {0, 100}},
			100,
		},
		{
			"rectangle 200x100",
			[4][2]float32{{0, 0}, {200, 0}, {200, 100}, {0, 100}},
			200,
		},
		{
			"zero",
			[4][2]float32{{0, 0}, {0, 0}, {0, 0}, {0, 0}},
			0,
		},
		{
			"tall 50x300",
			[4][2]float32{{0, 0}, {50, 0}, {50, 300}, {0, 300}},
			300,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := perspectiveSide(tt.quad); got != tt.want {
				t.Errorf("perspectiveSide() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestScaleLabel(t *testing.T) {
	tests := []struct {
		factor float64
		want   string
	}{
		{1.0, "1.00"},
		{0.5, "0.50"},
		{2.0, "2.00"},
		{0.25, "0.25"},
		{3.0, "3.00"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := scaleLabel(tt.factor); got != tt.want {
				t.Errorf("scaleLabel(%v) = %q, want %q", tt.factor, got, tt.want)
			}
		})
	}
}

func TestScaleImage(t *testing.T) {
	img := gocv.NewMatWithSize(100, 200, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	tests := []struct {
		name       string
		factor     float64
		wantW      int
		wantH      int
		wantEmpty  bool
	}{
		{"factor 1.0", 1.0, 200, 100, false},
		{"factor 2.0", 2.0, 400, 200, false},
		{"factor 0.5", 0.5, 100, 50, false},
		{"too small", 0.1, 0, 0, true},    // 200*0.1=20 < 25
		{"too large", 25.0, 0, 0, true},   // 200*25=5000 > 4096
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := scaleImage(img, tt.factor)
			defer result.Close()

			if tt.wantEmpty {
				if !result.Empty() {
					t.Errorf("Expected empty result for factor %v", tt.factor)
				}
				return
			}

			if result.Empty() {
				t.Fatalf("Got empty result for factor %v", tt.factor)
			}
			if result.Cols() != tt.wantW || result.Rows() != tt.wantH {
				t.Errorf("Dimensions = %dx%d, want %dx%d",
					result.Cols(), result.Rows(), tt.wantW, tt.wantH)
			}
		})
	}
}

func TestScaleImage_EmptyInput(t *testing.T) {
	emptyImg := gocv.NewMat()
	defer emptyImg.Close()

	result := scaleImage(emptyImg, 2.0)
	defer result.Close()

	// Empty input → should not crash (NewMat copy or empty)
	// The code does src.CopyTo for factor 1.0 but Resize for others.
	// For empty mat, Resize may produce empty result.
}

func TestCropBBox(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		BBoxXYXY: [4]float32{50, 50, 150, 150},
	}

	result := cropBBox(img, det)
	defer result.Close()

	if result.Empty() {
		t.Fatal("cropBBox returned empty mat")
	}
	if result.Cols() != 100 || result.Rows() != 100 {
		t.Errorf("Crop dimensions = %dx%d, want 100x100", result.Cols(), result.Rows())
	}
}

func TestCropBBox_DegenerateBBox(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()

	// x2 <= x1
	det := qrdet.Detection{
		BBoxXYXY: [4]float32{150, 50, 50, 150},
	}

	result := cropBBox(img, det)
	defer result.Close()

	if !result.Empty() {
		t.Error("Expected empty mat for degenerate bbox")
	}
}

func TestCropBBox_OutOfBounds(t *testing.T) {
	img := gocv.NewMatWithSize(200, 200, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	// BBox extends beyond image
	det := qrdet.Detection{
		BBoxXYXY: [4]float32{-50, -50, 250, 250},
	}

	result := cropBBox(img, det)
	defer result.Close()

	if result.Empty() {
		t.Fatal("cropBBox returned empty mat for out-of-bounds bbox")
	}
	// Should be clamped to image dimensions
	if result.Cols() != 200 || result.Rows() != 200 {
		t.Errorf("Crop dimensions = %dx%d, want 200x200 (clamped)", result.Cols(), result.Rows())
	}
}

func TestCropPerspective(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		PaddedQuadXY: [4][2]float32{
			{50, 50}, {250, 50}, {250, 250}, {50, 250},
		},
	}

	result := cropPerspective(img, det)
	defer result.Close()

	if result.Empty() {
		t.Fatal("cropPerspective returned empty mat")
	}
	// Perspective side = 200, output should be 200x200
	if result.Cols() != 200 || result.Rows() != 200 {
		t.Errorf("Crop dimensions = %dx%d, want 200x200", result.Cols(), result.Rows())
	}
}

func TestCropPerspective_ZeroQuad(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()

	det := qrdet.Detection{} // Zero quad → side < 1

	result := cropPerspective(img, det)
	defer result.Close()

	if !result.Empty() {
		t.Error("Expected empty mat for zero quad")
	}
}

func TestDeduplicateResults(t *testing.T) {
	tests := []struct {
		name    string
		results []QRResult
		want    int
	}{
		{
			"empty",
			nil,
			0,
		},
		{
			"single",
			[]QRResult{{Content: "a", Decoded: true, Detection: qrdet.Detection{Confidence: 0.9}}},
			1,
		},
		{
			"duplicate content keeps highest confidence",
			[]QRResult{
				{Content: "a", Decoded: true, Detection: qrdet.Detection{Confidence: 0.7}},
				{Content: "a", Decoded: true, Detection: qrdet.Detection{Confidence: 0.9}},
			},
			1,
		},
		{
			"different content kept",
			[]QRResult{
				{Content: "a", Decoded: true, Detection: qrdet.Detection{Confidence: 0.8}},
				{Content: "b", Decoded: true, Detection: qrdet.Detection{Confidence: 0.7}},
			},
			2,
		},
		{
			"undecoded always kept",
			[]QRResult{
				{Decoded: false, Detection: qrdet.Detection{Confidence: 0.5}},
				{Decoded: false, Detection: qrdet.Detection{Confidence: 0.3}},
			},
			2,
		},
		{
			"mix decoded and undecoded",
			[]QRResult{
				{Content: "a", Decoded: true, Detection: qrdet.Detection{Confidence: 0.8}},
				{Decoded: false, Detection: qrdet.Detection{Confidence: 0.5}},
				{Content: "a", Decoded: true, Detection: qrdet.Detection{Confidence: 0.9}},
				{Content: "b", Decoded: true, Detection: qrdet.Detection{Confidence: 0.6}},
			},
			3, // "a" deduped to 1, undecoded kept, "b" kept
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := deduplicateResults(tt.results)
			if len(got) != tt.want {
				t.Errorf("deduplicateResults() length = %d, want %d", len(got), tt.want)
			}
		})
	}
}

func TestDeduplicateResults_KeepsHigherConfidence(t *testing.T) {
	results := []QRResult{
		{Content: "test", Decoded: true, Detection: qrdet.Detection{Confidence: 0.5}},
		{Content: "test", Decoded: true, Detection: qrdet.Detection{Confidence: 0.9}},
		{Content: "test", Decoded: true, Detection: qrdet.Detection{Confidence: 0.7}},
	}

	deduped := deduplicateResults(results)
	if len(deduped) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(deduped))
	}
	if deduped[0].Detection.Confidence != 0.9 {
		t.Errorf("Kept confidence = %v, want 0.9", deduped[0].Detection.Confidence)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	expectedScales := []float64{0.5, 2, 0.25, 3, 4}
	if len(cfg.ScaleFactors) != len(expectedScales) {
		t.Fatalf("ScaleFactors length = %d, want %d", len(cfg.ScaleFactors), len(expectedScales))
	}
	for i, v := range cfg.ScaleFactors {
		if v != expectedScales[i] {
			t.Errorf("ScaleFactors[%d] = %v, want %v", i, v, expectedScales[i])
		}
	}

	if !cfg.TryPerspective {
		t.Error("TryPerspective should be true")
	}
	if cfg.MaxWorkers != 0 {
		t.Errorf("MaxWorkers = %d, want 0", cfg.MaxWorkers)
	}
	if cfg.DecoderFactory == nil {
		t.Error("DecoderFactory should not be nil")
	}
}

func TestMinimalConfig(t *testing.T) {
	cfg := MinimalConfig()

	if cfg.ScaleFactors != nil {
		t.Errorf("ScaleFactors = %v, want nil", cfg.ScaleFactors)
	}
	if cfg.TryPerspective {
		t.Error("TryPerspective should be false")
	}
	if cfg.DecoderFactory == nil {
		t.Error("DecoderFactory should not be nil")
	}
}

func TestTryDecode_EmptyImage(t *testing.T) {
	dec := &mockDecoder{results: nil, err: nil}

	emptyImg := gocv.NewMat()
	defer emptyImg.Close()

	content, strategy := tryDecode(dec, emptyImg, "test")
	if content != "" {
		t.Errorf("content = %q, want empty", content)
	}
	if strategy != "" {
		t.Errorf("strategy = %q, want empty", strategy)
	}
}

func TestTryDecode_Success(t *testing.T) {
	dec := &mockDecoder{
		results: []qrdecode.DecodedQR{{Content: "hello"}},
		err:     nil,
	}

	img := gocv.NewMatWithSize(100, 100, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	content, strategy := tryDecode(dec, img, "test_strategy")
	if content != "hello" {
		t.Errorf("content = %q, want %q", content, "hello")
	}
	if strategy != "test_strategy" {
		t.Errorf("strategy = %q, want %q", strategy, "test_strategy")
	}
}

func TestTryDecode_Error(t *testing.T) {
	dec := &mockDecoder{
		results: nil,
		err:     fmt.Errorf("decode error"),
	}

	img := gocv.NewMatWithSize(100, 100, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	content, _ := tryDecode(dec, img, "test")
	if content != "" {
		t.Errorf("content = %q, want empty on error", content)
	}
}

func TestTryDecode_EmptyResults(t *testing.T) {
	dec := &mockDecoder{
		results: []qrdecode.DecodedQR{},
		err:     nil,
	}

	img := gocv.NewMatWithSize(100, 100, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	content, _ := tryDecode(dec, img, "test")
	if content != "" {
		t.Errorf("content = %q, want empty for no results", content)
	}
}

func TestDecodeDetection_DirectDecode(t *testing.T) {
	// Create image with some content
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		Confidence: 0.9,
		BBoxXYXY:   [4]float32{50, 50, 350, 350},
		PaddedQuadXY: [4][2]float32{
			{50, 50}, {350, 50}, {350, 350}, {50, 350},
		},
	}

	// Mock decoder that succeeds on first try
	dec := &mockDecoder{
		results: []qrdecode.DecodedQR{{Content: "HELLO"}},
	}

	cfg := Config{
		ScaleFactors:   []float64{0.5, 2.0},
		TryPerspective: true,
	}

	result := decodeDetection(img, det, dec, cfg)
	if !result.Decoded {
		t.Error("Expected decoded=true")
	}
	if result.Content != "HELLO" {
		t.Errorf("Content = %q, want %q", result.Content, "HELLO")
	}
	if result.Strategy == "" {
		t.Error("Strategy should not be empty")
	}
}

func TestDecodeDetection_NoQR(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		Confidence: 0.9,
		BBoxXYXY:   [4]float32{50, 50, 350, 350},
		PaddedQuadXY: [4][2]float32{
			{50, 50}, {350, 50}, {350, 350}, {50, 350},
		},
	}

	// Mock decoder that never finds a QR code
	dec := &mockDecoder{results: nil, err: nil}

	cfg := Config{
		ScaleFactors:   []float64{0.5},
		TryPerspective: false,
	}

	result := decodeDetection(img, det, dec, cfg)
	if result.Decoded {
		t.Error("Expected decoded=false when no QR found")
	}
	if result.Content != "" {
		t.Errorf("Content = %q, want empty", result.Content)
	}
}

func TestDecodeDetection_NoPerspective(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		BBoxXYXY: [4]float32{50, 50, 350, 350},
	}

	dec := &mockDecoder{results: nil}

	cfg := Config{
		ScaleFactors:   nil,
		TryPerspective: false,
	}

	result := decodeDetection(img, det, dec, cfg)
	if result.Decoded {
		t.Error("Expected decoded=false")
	}
}

// mockDecoderFactory returns a factory and a pointer to a call counter.
func mockDecoderFactory(results []qrdecode.DecodedQR) (func() (qrdecode.Decoder, error), *int) {
	calls := new(int)
	return func() (qrdecode.Decoder, error) {
		*calls++
		return &mockDecoder{results: results}, nil
	}, calls
}

func TestRetryWithFallbacks_FallbackSucceeds(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		Confidence: 0.9,
		BBoxXYXY:   [4]float32{50, 50, 350, 350},
	}
	failed := QRResult{Detection: det, Decoded: false}

	fallbackFactory, calls := mockDecoderFactory([]qrdecode.DecodedQR{{Content: "FALLBACK"}})

	q := &QReader{
		config: Config{
			TryHarder:                true,
			FallbackDecoderFactories: []func() (qrdecode.Decoder, error){fallbackFactory},
			ScaleFactors:             []float64{1.0},
			TryPerspective:           false,
		},
	}

	result := q.retryWithFallbacks(img, det, failed)

	if !result.Decoded {
		t.Error("Expected fallback to decode successfully")
	}
	if result.Content != "FALLBACK" {
		t.Errorf("Content = %q, want %q", result.Content, "FALLBACK")
	}
	if *calls != 1 {
		t.Errorf("fallback factory called %d times, want 1", *calls)
	}
}

func TestRetryWithFallbacks_StopsOnFirstSuccess(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		Confidence: 0.9,
		BBoxXYXY:   [4]float32{50, 50, 350, 350},
	}
	failed := QRResult{Detection: det, Decoded: false}

	firstFactory, firstCalls := mockDecoderFactory([]qrdecode.DecodedQR{{Content: "FIRST"}})
	secondFactory, secondCalls := mockDecoderFactory([]qrdecode.DecodedQR{{Content: "SECOND"}})

	q := &QReader{
		config: Config{
			TryHarder:                true,
			FallbackDecoderFactories: []func() (qrdecode.Decoder, error){firstFactory, secondFactory},
			ScaleFactors:             []float64{1.0},
			TryPerspective:           false,
		},
	}

	result := q.retryWithFallbacks(img, det, failed)

	if result.Content != "FIRST" {
		t.Errorf("Content = %q, want FIRST (should stop at first success)", result.Content)
	}
	if *firstCalls != 1 {
		t.Errorf("first factory called %d times, want 1", *firstCalls)
	}
	if *secondCalls != 0 {
		t.Errorf("second factory called %d times, want 0 (should not be reached)", *secondCalls)
	}
}

func TestRetryWithFallbacks_AllFail(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		Confidence: 0.9,
		BBoxXYXY:   [4]float32{50, 50, 350, 350},
	}
	failed := QRResult{Detection: det, Decoded: false}

	// All fallback decoders return nothing
	emptyFactory, _ := mockDecoderFactory(nil)

	q := &QReader{
		config: Config{
			TryHarder:                true,
			FallbackDecoderFactories: []func() (qrdecode.Decoder, error){emptyFactory, emptyFactory},
			ScaleFactors:             []float64{1.0},
			TryPerspective:           false,
		},
	}

	result := q.retryWithFallbacks(img, det, failed)

	if result.Decoded {
		t.Error("Expected decoded=false when all fallbacks fail")
	}
}

// TestRetryWithFallbacks_FallbackOverridesMisEncodedPrimary verifies that the
// fallback is run even when the primary decoded successfully, so that a decoder
// with better encoding support (e.g. OpenCV) can override a mis-encoded ZBar
// result (e.g. Shift-JIS conversion artifacts).
func TestRetryWithFallbacks_FallbackOverridesMisEncodedPrimary(t *testing.T) {
	img := gocv.NewMatWithSize(400, 400, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(128, 128, 128, 0))

	det := qrdet.Detection{
		Confidence: 0.9,
		BBoxXYXY:   [4]float32{50, 50, 350, 350},
	}

	// Simulate ZBar returning a decoded result that is valid UTF-8 but with wrong
	// characters due to Shift-JIS misidentification (獺 instead of ál).
	misEncodedResult := QRResult{
		Detection: det,
		Decoded:   true,
		Content:   "Som sexu\u7b7alny", // 獺 is U+7B7A, a common Shift-JIS artifact
	}

	fallbackFactory, calls := mockDecoderFactory([]qrdecode.DecodedQR{{Content: "Som sexuálny"}})

	q := &QReader{
		config: Config{
			TryHarder:                true,
			FallbackDecoderFactories: []func() (qrdecode.Decoder, error){fallbackFactory},
			ScaleFactors:             []float64{1.0},
			TryPerspective:           false,
		},
	}

	result := q.retryWithFallbacks(img, det, misEncodedResult)

	if result.Content != "Som sexuálny" {
		t.Errorf("Content = %q, want correct UTF-8 from fallback", result.Content)
	}
	if *calls != 1 {
		t.Errorf("fallback factory called %d times, want 1", *calls)
	}
}

func TestRetryWithFallbacks_NoFactories(t *testing.T) {
	img := gocv.NewMatWithSize(100, 100, gocv.MatTypeCV8UC3)
	defer img.Close()

	det := qrdet.Detection{}
	failed := QRResult{Detection: det, Decoded: false}

	q := &QReader{
		config: Config{
			TryHarder:                true,
			FallbackDecoderFactories: nil,
		},
	}

	result := q.retryWithFallbacks(img, det, failed)

	if result.Decoded {
		t.Error("Expected decoded=false with no factories")
	}
}

func TestQRResult_Struct(t *testing.T) {
	r := QRResult{
		Detection: qrdet.Detection{Confidence: 0.95},
		Content:   "https://example.com",
		Decoded:   true,
		Strategy:  "bbox/scale_2.00/inversion",
	}

	if r.Detection.Confidence != 0.95 {
		t.Errorf("Detection.Confidence = %v, want 0.95", r.Detection.Confidence)
	}
	if r.Content != "https://example.com" {
		t.Errorf("Content = %q, want %q", r.Content, "https://example.com")
	}
	if !r.Decoded {
		t.Error("Decoded should be true")
	}
	if r.Strategy != "bbox/scale_2.00/inversion" {
		t.Errorf("Strategy = %q, want %q", r.Strategy, "bbox/scale_2.00/inversion")
	}
}
