package qrdet

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"

	"gocv.io/x/gocv"
)

func TestNewImageFromMat(t *testing.T) {
	mat := gocv.NewMatWithSize(100, 200, gocv.MatTypeCV8UC3)
	defer mat.Close()
	mat.SetTo(gocv.NewScalar(128, 128, 128, 0))

	img := NewImageFromMat(mat)

	if img.Width() != 200 {
		t.Errorf("Width() = %d, want 200", img.Width())
	}
	if img.Height() != 100 {
		t.Errorf("Height() = %d, want 100", img.Height())
	}
	m := img.MatRef()
	if m.Empty() {
		t.Error("MatRef() is empty")
	}
}

func TestNewImageFromRaw(t *testing.T) {
	width, height := 100, 50
	data := make([]byte, width*height*3)
	for i := 0; i < len(data); i += 3 {
		data[i] = 255   // B
		data[i+1] = 128 // G
		data[i+2] = 64  // R
	}

	img, err := NewImageFromRaw(data, width, height)
	if err != nil {
		t.Fatalf("NewImageFromRaw() error = %v", err)
	}
	defer img.Close()

	if img.Width() != width {
		t.Errorf("Width() = %d, want %d", img.Width(), width)
	}
	if img.Height() != height {
		t.Errorf("Height() = %d, want %d", img.Height(), height)
	}
}

func TestNewImageFromRaw_Errors(t *testing.T) {
	tests := []struct {
		name   string
		data   []byte
		width  int
		height int
	}{
		{"wrong size", make([]byte, 100), 100, 50},
		{"zero width", make([]byte, 0), 0, 50},
		{"zero height", make([]byte, 0), 100, 0},
		{"negative width", nil, -1, 50},
		{"negative height", nil, 100, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewImageFromRaw(tt.data, tt.width, tt.height)
			if err == nil {
				t.Error("NewImageFromRaw() expected error, got nil")
			}
		})
	}
}

func TestNewImageFromBytes(t *testing.T) {
	// Create a valid PNG in memory
	img := image.NewNRGBA(image.Rect(0, 0, 50, 30))
	for y := 0; y < 30; y++ {
		for x := 0; x < 50; x++ {
			img.Set(x, y, color.NRGBA{R: 200, G: 150, B: 100, A: 255})
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("Failed to encode PNG: %v", err)
	}

	src, err := NewImageFromBytes(buf.Bytes(), ImageFormatPNG)
	if err != nil {
		t.Fatalf("NewImageFromBytes() error = %v", err)
	}
	defer src.Close()

	if src.Width() != 50 {
		t.Errorf("Width() = %d, want 50", src.Width())
	}
	if src.Height() != 30 {
		t.Errorf("Height() = %d, want 30", src.Height())
	}
}

func TestNewImageFromBytes_Auto(t *testing.T) {
	// Also test with ImageFormatAuto
	img := image.NewNRGBA(image.Rect(0, 0, 10, 10))
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("Failed to encode PNG: %v", err)
	}

	src, err := NewImageFromBytes(buf.Bytes(), ImageFormatAuto)
	if err != nil {
		t.Fatalf("NewImageFromBytes(Auto) error = %v", err)
	}
	defer src.Close()

	if src.Width() != 10 || src.Height() != 10 {
		t.Errorf("Dimensions = %dx%d, want 10x10", src.Width(), src.Height())
	}
}

func TestNewImageFromBytes_Errors(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"empty data", nil},
		{"empty slice", []byte{}},
		{"invalid data", []byte{0x00, 0x01, 0x02, 0x03}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewImageFromBytes(tt.data, ImageFormatAuto)
			if err == nil {
				t.Error("NewImageFromBytes() expected error, got nil")
			}
		})
	}
}

func TestNewImageFromFile_NonExistent(t *testing.T) {
	_, err := NewImageFromFile("/nonexistent/path/image.jpg")
	if err == nil {
		t.Error("NewImageFromFile() with nonexistent path expected error, got nil")
	}
}

func TestImageSource_Close_Nil(t *testing.T) {
	var img *ImageSource
	// Should not panic
	err := img.Close()
	if err != nil {
		t.Errorf("Close() on nil ImageSource returned error: %v", err)
	}
}

func TestImageSource_Close_Empty(t *testing.T) {
	emptyMat := gocv.NewMat()
	defer emptyMat.Close()

	img := NewImageFromMat(emptyMat)
	err := img.Close()
	if err != nil {
		t.Errorf("Close() on empty ImageSource returned error: %v", err)
	}
}

func TestImageSource_WidthHeight_Nil(t *testing.T) {
	var img *ImageSource
	if img.Width() != 0 {
		t.Errorf("Width() on nil = %d, want 0", img.Width())
	}
	if img.Height() != 0 {
		t.Errorf("Height() on nil = %d, want 0", img.Height())
	}
}

func TestImageSource_WidthHeight_Empty(t *testing.T) {
	emptyMat := gocv.NewMat()
	defer emptyMat.Close()

	img := NewImageFromMat(emptyMat)
	if img.Width() != 0 {
		t.Errorf("Width() on empty = %d, want 0", img.Width())
	}
	if img.Height() != 0 {
		t.Errorf("Height() on empty = %d, want 0", img.Height())
	}
}

func TestImageFormatConstants(t *testing.T) {
	// Verify format enum values are distinct
	formats := []ImageFormat{ImageFormatAuto, ImageFormatJPEG, ImageFormatPNG, ImageFormatBMP}
	seen := make(map[ImageFormat]bool)
	for _, f := range formats {
		if seen[f] {
			t.Errorf("Duplicate ImageFormat value: %d", f)
		}
		seen[f] = true
	}
}
