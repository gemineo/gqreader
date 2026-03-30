package qrdecode

import (
	"testing"

	"gocv.io/x/gocv"
)

func TestNewOpenCVDecoder(t *testing.T) {
	dec, err := NewOpenCVDecoder()
	if err != nil {
		t.Fatalf("NewOpenCVDecoder() error = %v", err)
	}
	defer dec.Close()
}

func TestOpenCVDecoder_Decode_EmptyImage(t *testing.T) {
	dec, err := NewOpenCVDecoder()
	if err != nil {
		t.Fatalf("NewOpenCVDecoder() error = %v", err)
	}
	defer dec.Close()

	emptyImg := gocv.NewMat()
	defer emptyImg.Close()

	_, err = dec.Decode(emptyImg)
	if err == nil {
		t.Error("Decode(empty) should return error")
	}
}

func TestOpenCVDecoder_Decode_NoQR(t *testing.T) {
	dec, err := NewOpenCVDecoder()
	if err != nil {
		t.Fatalf("NewOpenCVDecoder() error = %v", err)
	}
	defer dec.Close()

	// Plain white image - no QR code
	img := gocv.NewMatWithSize(200, 200, gocv.MatTypeCV8UC3)
	defer img.Close()
	img.SetTo(gocv.NewScalar(255, 255, 255, 0))

	results, err := dec.Decode(img)
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Decode(white image) returned %d results, want 0", len(results))
	}
}

func TestOpenCVDecoder_Close(t *testing.T) {
	dec, err := NewOpenCVDecoder()
	if err != nil {
		t.Fatalf("NewOpenCVDecoder() error = %v", err)
	}

	err = dec.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}
}

func TestOpenCVDecoder_ImplementsDecoder(t *testing.T) {
	dec, err := NewOpenCVDecoder()
	if err != nil {
		t.Fatalf("NewOpenCVDecoder() error = %v", err)
	}
	defer dec.Close()

	// Runtime interface check
	var _ Decoder = dec
}

func TestDecodedQR_Struct(t *testing.T) {
	qr := DecodedQR{
		Content: "hello world",
		Points: [][2]float32{
			{10, 20}, {110, 20}, {110, 120}, {10, 120},
		},
	}

	if qr.Content != "hello world" {
		t.Errorf("Content = %q, want %q", qr.Content, "hello world")
	}
	if len(qr.Points) != 4 {
		t.Errorf("Points length = %d, want 4", len(qr.Points))
	}
}

func TestDecodedQR_NilPoints(t *testing.T) {
	qr := DecodedQR{Content: "test"}
	if qr.Points != nil {
		t.Errorf("Points should be nil, got %v", qr.Points)
	}
}
