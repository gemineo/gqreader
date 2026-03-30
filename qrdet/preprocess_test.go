package qrdet

import (
	"testing"

	"gocv.io/x/gocv"
)

func TestPreprocessImage(t *testing.T) {
	// Create a test image (100x200, RGB)
	testImg := gocv.NewMatWithSize(200, 100, gocv.MatTypeCV8UC3)
	defer testImg.Close()

	// Fill with some test pattern
	testImg.SetTo(gocv.NewScalar(128, 128, 128, 0))

	// Test preprocessing
	inputSize := 640
	data, info, err := preprocessImage(testImg, inputSize)

	if err != nil {
		t.Fatalf("preprocessImage() error = %v", err)
	}

	// Verify output data length
	expectedLen := 3 * inputSize * inputSize
	if len(data) != expectedLen {
		t.Errorf("preprocessImage() data length = %d, want %d", len(data), expectedLen)
	}

	// Verify data values are normalized [0, 1]
	for i, v := range data {
		if v < 0 || v > 1 {
			t.Errorf("preprocessImage() data[%d] = %v, outside [0,1] range", i, v)
			break
		}
	}

	// Verify letterbox info is valid
	if info.scale <= 0 {
		t.Errorf("preprocessImage() invalid scale: scale=%v", info.scale)
	}

	if info.padLeft < 0 || info.padTop < 0 {
		t.Errorf("preprocessImage() invalid padding: padLeft=%v, padTop=%v", info.padLeft, info.padTop)
	}

	if info.originalWidth != testImg.Cols() || info.originalHeight != testImg.Rows() {
		t.Errorf("preprocessImage() incorrect original dimensions: got (%v,%v), want (%v,%v)",
			info.originalWidth, info.originalHeight, testImg.Cols(), testImg.Rows())
	}

	if info.targetSize != inputSize {
		t.Errorf("preprocessImage() targetSize = %v, want %v", info.targetSize, inputSize)
	}
}

func TestPreprocessImage_Square(t *testing.T) {
	// Test with square image (no letterboxing needed)
	testImg := gocv.NewMatWithSize(640, 640, gocv.MatTypeCV8UC3)
	defer testImg.Close()

	testImg.SetTo(gocv.NewScalar(100, 150, 200, 0))

	data, info, err := preprocessImage(testImg, 640)

	if err != nil {
		t.Fatalf("preprocessImage() error = %v", err)
	}

	// For square image matching input size, scale should be 1.0 and padding should be 0
	tolerance := float32(0.01)
	if info.scale < 1.0-tolerance || info.scale > 1.0+tolerance {
		t.Errorf("preprocessImage() scale = %v, want 1.0 for square image", info.scale)
	}

	if info.padLeft != 0 || info.padTop != 0 {
		t.Errorf("preprocessImage() padding = (%v, %v), want (0, 0) for square image", info.padLeft, info.padTop)
	}

	// Verify output length
	expectedLen := 3 * 640 * 640
	if len(data) != expectedLen {
		t.Errorf("preprocessImage() data length = %d, want %d", len(data), expectedLen)
	}
}

func TestPreprocessImage_EmptyImage(t *testing.T) {
	// Test with empty Mat
	emptyImg := gocv.NewMat()
	defer emptyImg.Close()

	_, _, err := preprocessImage(emptyImg, 640)

	if err == nil {
		t.Error("preprocessImage() with empty image should return error")
	}
}

func TestPreprocessImage_DifferentSizes(t *testing.T) {
	tests := []struct {
		name       string
		width      int
		height     int
		targetSize int
	}{
		{"landscape 4:3", 800, 600, 640},
		{"portrait 3:4", 600, 800, 640},
		{"very wide", 1920, 480, 640},
		{"very tall", 480, 1920, 640},
		{"small square", 100, 100, 640},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img := gocv.NewMatWithSize(tt.height, tt.width, gocv.MatTypeCV8UC3)
			defer img.Close()

			img.SetTo(gocv.NewScalar(128, 128, 128, 0))

			data, info, err := preprocessImage(img, tt.targetSize)

			if err != nil {
				t.Fatalf("preprocessImage() error = %v", err)
			}

			// Verify output size
			expectedLen := 3 * tt.targetSize * tt.targetSize
			if len(data) != expectedLen {
				t.Errorf("data length = %d, want %d", len(data), expectedLen)
			}

			// Verify info
			if info.originalWidth != tt.width {
				t.Errorf("originalWidth = %d, want %d", info.originalWidth, tt.width)
			}

			if info.originalHeight != tt.height {
				t.Errorf("originalHeight = %d, want %d", info.originalHeight, tt.height)
			}

			if info.targetSize != tt.targetSize {
				t.Errorf("targetSize = %d, want %d", info.targetSize, tt.targetSize)
			}

			// Verify scale is positive
			if info.scale <= 0 {
				t.Errorf("scale = %v, should be > 0", info.scale)
			}

			// Verify at most one padding dimension is non-zero
			if info.padLeft != 0 && info.padTop != 0 {
				t.Errorf("both padLeft (%d) and padTop (%d) are non-zero, expected at most one",
					info.padLeft, info.padTop)
			}
		})
	}
}

func TestMatToCHW(t *testing.T) {
	// Create a small test image (3x3 RGB)
	testImg := gocv.NewMatWithSize(3, 3, gocv.MatTypeCV8UC3)
	defer testImg.Close()

	// Set specific values for verification (BGR order: B=100, G=150, R=200)
	testImg.SetTo(gocv.NewScalar(100, 150, 200, 0))

	// Convert to float32 (matToCHW expects CV32FC3)
	floatImg := gocv.NewMat()
	defer floatImg.Close()
	testImg.ConvertTo(&floatImg, gocv.MatTypeCV32FC3)
	floatImg.DivideFloat(255.0) // Normalize to [0, 1]

	result, err := matToCHW(floatImg)
	if err != nil {
		t.Fatalf("matToCHW() error = %v", err)
	}

	// Verify output length
	expectedLen := 3 * 3 * 3 // channels * height * width
	if len(result) != expectedLen {
		t.Fatalf("matToCHW() length = %d, want %d", len(result), expectedLen)
	}

	// Verify normalization is applied
	// All values should be in [0, 1] range
	for i, v := range result {
		if v < 0 || v > 1 {
			t.Errorf("matToCHW()[%d] = %v, outside [0,1] range", i, v)
			break
		}
	}

	// Verify CHW layout: all pixels should have same normalized values
	// Note: matToCHW does NOT do BGR->RGB conversion, it just rearranges to CHW
	// Input is BGR: B=100, G=150, R=200 (from gocv.Scalar)
	// Expected normalized values (still in BGR order):
	// Channel 0 (B): 100/255 ≈ 0.392
	// Channel 1 (G): 150/255 ≈ 0.588
	// Channel 2 (R): 200/255 ≈ 0.784
	expectedB := float32(100) / 255.0
	expectedG := float32(150) / 255.0
	expectedR := float32(200) / 255.0

	tolerance := float32(0.01)

	// Check first pixel values in CHW order (BGR)
	b := result[0]     // First channel (B), first pixel
	g := result[1*3*3] // Second channel (G), first pixel
	r := result[2*3*3] // Third channel (R), first pixel

	if b < expectedB-tolerance || b > expectedB+tolerance {
		t.Errorf("matToCHW() B channel = %v, want %v", b, expectedB)
	}

	if g < expectedG-tolerance || g > expectedG+tolerance {
		t.Errorf("matToCHW() G channel = %v, want %v", g, expectedG)
	}

	if r < expectedR-tolerance || r > expectedR+tolerance {
		t.Errorf("matToCHW() R channel = %v, want %v", r, expectedR)
	}
}

func TestMatToCHW_Empty(t *testing.T) {
	emptyImg := gocv.NewMat()
	defer emptyImg.Close()

	result, err := matToCHW(emptyImg)

	if err == nil {
		t.Error("matToCHW() with empty image should return error")
	}

	if result != nil {
		t.Errorf("matToCHW() with empty image returned %d elements, want nil", len(result))
	}
}

func TestMatToCHW_Normalization(t *testing.T) {
	// Test that all values are properly normalized to [0, 1]
	testImg := gocv.NewMatWithSize(10, 10, gocv.MatTypeCV8UC3)
	defer testImg.Close()

	// Set all pixels to max value (255)
	testImg.SetTo(gocv.NewScalar(255, 255, 255, 0))

	// Convert to float32 (matToCHW expects CV32FC3)
	floatImg := gocv.NewMat()
	defer floatImg.Close()
	testImg.ConvertTo(&floatImg, gocv.MatTypeCV32FC3)
	floatImg.DivideFloat(255.0) // Normalize to [0, 1]

	result, err := matToCHW(floatImg)
	if err != nil {
		t.Fatalf("matToCHW() error = %v", err)
	}

	// All values should be close to 1.0
	for i, v := range result {
		if v < 0.99 || v > 1.01 {
			t.Errorf("matToCHW()[%d] = %v, expected ~1.0 for max input value", i, v)
			break
		}
	}

	// Test with min value (0)
	testImg.SetTo(gocv.NewScalar(0, 0, 0, 0))
	testImg.ConvertTo(&floatImg, gocv.MatTypeCV32FC3)
	floatImg.DivideFloat(255.0)

	result, err = matToCHW(floatImg)
	if err != nil {
		t.Fatalf("matToCHW() error = %v", err)
	}

	// All values should be 0.0
	for i, v := range result {
		if v < -0.01 || v > 0.01 {
			t.Errorf("matToCHW()[%d] = %v, expected ~0.0 for min input value", i, v)
			break
		}
	}
}

// Benchmark tests
func BenchmarkPreprocessImage(b *testing.B) {
	testImg := gocv.NewMatWithSize(480, 640, gocv.MatTypeCV8UC3)
	defer testImg.Close()

	testImg.SetTo(gocv.NewScalar(128, 128, 128, 0))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = preprocessImage(testImg, 640)
	}
}

func BenchmarkMatToCHW(b *testing.B) {
	testImg := gocv.NewMatWithSize(640, 640, gocv.MatTypeCV8UC3)
	defer testImg.Close()

	testImg.SetTo(gocv.NewScalar(128, 128, 128, 0))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = matToCHW(testImg)
	}
}
