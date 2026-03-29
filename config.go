package gqreader

import (
	"github.com/gemineo/gqreader/qrdecode"
	"github.com/gemineo/gqreader/qrdet"
)

// Config holds configuration for the QReader pipeline.
type Config struct {
	// DetectionConfig configures the YOLOv8 detector.
	DetectionConfig qrdet.Config

	// ModelPath is the path to the ONNX model file.
	// If empty, the embedded model is used (requires embed_model build tag).
	ModelPath string

	// ScaleFactors lists the scale factors to try during decoding.
	// Default: {0.5, 2, 0.25, 3, 4} (matching Python QReader).
	// Set to nil or empty to skip scaling.
	ScaleFactors []float64

	// TryPerspective enables perspective-corrected cropping in addition to bbox cropping.
	// Default: true.
	TryPerspective bool

	// MaxWorkers controls parallelism for processing multiple detections.
	// 0 means one goroutine per detection (unlimited).
	// Default: 0.
	MaxWorkers int

	// DecoderFactory creates a new Decoder instance.
	// Default: OpenCV QRCodeDetector.
	DecoderFactory func() (qrdecode.Decoder, error)
}

// DefaultConfig returns the default configuration matching Python QReader's full pipeline.
// Uses OpenCV QRCodeDetector for maximum portability.
// For better performance, use ZBarConfig() if libzbar is available.
func DefaultConfig() Config {
	return Config{
		DetectionConfig: qrdet.DefaultConfig(),
		ScaleFactors:    []float64{0.5, 2, 0.25, 3, 4},
		TryPerspective:  true,
		MaxWorkers:      0,
		DecoderFactory:  func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() },
	}
}

// ZBarConfig returns the default pipeline using libzbar as the QR decoder.
// ZBar is typically 3-5× faster than OpenCV QRCodeDetector and is the same
// library used by Python QReader (pyzbar). Requires libzbar:
//
//	macOS:  brew install zbar
//	Linux:  apt-get install libzbar-dev
func ZBarConfig() Config {
	return Config{
		DetectionConfig: qrdet.DefaultConfig(),
		ScaleFactors:    []float64{0.5, 2, 0.25, 3, 4},
		TryPerspective:  true,
		MaxWorkers:      0,
		DecoderFactory:  func() (qrdecode.Decoder, error) { return qrdecode.NewZBarDecoder() },
	}
}

// MinimalConfig returns a fast configuration that skips scaling and perspective correction.
// Useful for high-quality images where QR codes are easy to decode.
func MinimalConfig() Config {
	return Config{
		DetectionConfig: qrdet.DefaultConfig(),
		ScaleFactors:    nil,
		TryPerspective:  false,
		MaxWorkers:      0,
		DecoderFactory:  func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() },
	}
}
