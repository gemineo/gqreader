package gqreader

import (
	"github.com/gemineo/gqreader/qrdecode"
	"github.com/gemineo/gqreader/qrdet"
	"github.com/gemineo/gqreader/qrstruct"
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

	// TryHarder enables the fallback decoder chain when the primary decoder
	// fails all Phase 1+2 preprocessing attempts.
	TryHarder bool

	// FallbackDecoderFactories is an ordered list of decoder factories tried in
	// sequence when TryHarder is true and the primary decoder fails. Each factory
	// must produce a fresh, independent Decoder instance. Nil means no fallback.
	FallbackDecoderFactories []func() (qrdecode.Decoder, error)

	// DecoderFactory creates a new Decoder instance.
	// When MaxWorkers > 1, the pipeline calls this factory once per goroutine
	// so each worker owns its own decoder. Implementations need not be
	// concurrent-safe, but must be safe for sequential reuse within a single goroutine.
	// Default: OpenCV QRCodeDetector.
	DecoderFactory func() (qrdecode.Decoder, error)

	// TryStructured enables Phase 3: when every decoder and preprocessing
	// strategy has failed on a detection, a sampling grid is fitted to the
	// detection and the module matrix is decoded directly (package qrstruct).
	// This recovers low-resolution and curved symbols (cups, bottles) and
	// only runs on detections that would otherwise be reported undecoded.
	TryStructured bool

	// StructuredOptions tunes Phase 3. The zero value selects
	// qrstruct.DefaultOptions().
	StructuredOptions qrstruct.Options
}

// DefaultConfig returns the recommended configuration: ZBar primary decoder with
// OpenCV as the fallback, then the structured decoder as a last resort.
// Achieves 100% image accuracy on Dataset-QR-2020, exceeding Python QReader.
// Use FastConfig() when throughput matters more than decode accuracy.
//
// Requires libzbar (brew install zbar / apt-get install libzbar-dev).
func DefaultConfig() Config {
	return FastConfig().WithFallbackDecoderFactories(
		func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() },
	).WithStructured(true)
}

// FastConfig returns a speed-optimised configuration: ZBar decoder only, no fallback.
// Suitable when throughput matters more than decode accuracy on encoding-ambiguous QR codes.
// Known limitation: ZBar may mis-encode ISO-8859-1 byte sequences (e.g. Latin-1 text);
// use DefaultConfig() if QR codes may contain non-ASCII Latin characters.
//
// Requires libzbar (brew install zbar / apt-get install libzbar-dev).
func FastConfig() Config {
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

// WithFallbackDecoderFactories returns a copy of the config with TryHarder enabled
// and the given factories appended to any existing FallbackDecoderFactories.
// Factories are tried in order when the primary decoder fails all Phase 1+2 attempts.
func (c Config) WithFallbackDecoderFactories(fs ...func() (qrdecode.Decoder, error)) Config {
	newSlice := make([]func() (qrdecode.Decoder, error), len(c.FallbackDecoderFactories)+len(fs))
	copy(newSlice, c.FallbackDecoderFactories)
	copy(newSlice[len(c.FallbackDecoderFactories):], fs)
	c.FallbackDecoderFactories = newSlice
	c.TryHarder = true
	return c
}

// WithStructured returns a copy of the config with Phase 3 (structured decoding)
// enabled or disabled.
func (c Config) WithStructured(v bool) Config {
	c.TryStructured = v
	return c
}

// WithStructuredOptions returns a copy of the config with Phase 3 enabled and
// tuned by the given options.
func (c Config) WithStructuredOptions(opts qrstruct.Options) Config {
	c.TryStructured = true
	c.StructuredOptions = opts
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
