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
	// When MaxWorkers > 1, the pipeline calls this factory once per goroutine
	// so each worker owns its own decoder. Implementations need not be
	// concurrent-safe, but must be safe for sequential reuse within a single goroutine.
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

// WithDetectionConfig returns a copy of the config with the given detection config.
func (c Config) WithDetectionConfig(cfg qrdet.Config) Config {
	c.DetectionConfig = cfg
	return c
}

// WithScaleFactors returns a copy of the config with the given scale factors.
// Pass nil or no arguments to disable scaling (equivalent to MinimalConfig scaling behaviour).
func (c Config) WithScaleFactors(factors ...float64) Config {
	if len(factors) == 0 {
		c.ScaleFactors = nil
	} else {
		c.ScaleFactors = factors
	}
	return c
}

// WithTryPerspective returns a copy of the config with perspective-corrected cropping enabled or disabled.
func (c Config) WithTryPerspective(v bool) Config {
	c.TryPerspective = v
	return c
}

// WithMaxWorkers returns a copy of the config with the given maximum worker count.
// 0 means one goroutine per detection (unbounded).
func (c Config) WithMaxWorkers(n int) Config {
	c.MaxWorkers = n
	return c
}

// WithDecoderFactory returns a copy of the config with the given decoder factory.
// The factory must produce a new, independent Decoder instance on each call.
func (c Config) WithDecoderFactory(f func() (qrdecode.Decoder, error)) Config {
	c.DecoderFactory = f
	return c
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
