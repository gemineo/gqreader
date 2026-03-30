package imgprep

import (
	"iter"

	"gocv.io/x/gocv"
)

// DefaultStrategies returns the recommended set of preprocessing strategies
// for general document processing. Strategies are ordered by complexity,
// from fast to thorough.
func DefaultStrategies() []Strategy {
	return []Strategy{
		FastStrategy(),
		StandardStrategy(),
		DegradedStrategy(),
		FaxStrategy(),
	}
}

// AllStrategies returns all available preprocessing strategies,
// including rotation search. Use this when recall is more important
// than processing time.
func AllStrategies() []Strategy {
	return []Strategy{
		FastStrategy(),
		StandardStrategy(),
		DegradedStrategy(),
		FaxStrategy(),
		HighContrastStrategy(),
		SauvolaStrategy(),
		RotationStrategy(),
	}
}

// FastStrategy returns a minimal preprocessing pipeline.
// Best for high-quality scans with good lighting.
//
// Pipeline: Grayscale → Gaussian(3) → Otsu
func FastStrategy() Strategy {
	return NewCompositeStrategy("fast",
		&Grayscale{},
		&GaussianBlur{KernelSize: 3},
		&OtsuThreshold{},
	)
}

// StandardStrategy returns the recommended default pipeline.
// Good balance between quality and speed for typical office documents.
//
// Pipeline: Grayscale → CLAHE(2.0) → AdaptiveThreshold(Gaussian, 11, 2)
func StandardStrategy() Strategy {
	return NewCompositeStrategy("standard",
		&Grayscale{},
		&CLAHE{ClipLimit: 2.0, TileSize: 8},
		&AdaptiveThreshold{
			BlockSize: 11,
			C:         2,
			Method:    gocv.AdaptiveThresholdGaussian,
		},
	)
}

// DegradedStrategy returns a pipeline optimized for degraded documents.
// Use for old documents, multiple-generation copies, or low contrast.
//
// Pipeline: Grayscale → Median(5) → CLAHE(3.0) → AdaptiveThreshold(Gaussian, 31, 5)
func DegradedStrategy() Strategy {
	return NewCompositeStrategy("degraded",
		&Grayscale{},
		&MedianBlur{KernelSize: 5},
		&CLAHE{ClipLimit: 3.0, TileSize: 8},
		&AdaptiveThreshold{
			BlockSize: 31,
			C:         5,
			Method:    gocv.AdaptiveThresholdGaussian,
		},
	)
}

// FaxStrategy returns a pipeline optimized for fax documents.
// Handles the characteristic salt-and-pepper noise and low resolution.
//
// Pipeline: Grayscale → Median(3) → Median(3) → CLAHE(2.5) → AdaptiveThreshold → MorphOpen(2)
func FaxStrategy() Strategy {
	return NewCompositeStrategy("fax",
		&Grayscale{},
		&MedianBlur{KernelSize: 3},
		&MedianBlur{KernelSize: 3}, // Double median is very effective for fax noise
		&CLAHE{ClipLimit: 2.5, TileSize: 8},
		&AdaptiveThreshold{
			BlockSize: 15,
			C:         3,
			Method:    gocv.AdaptiveThresholdGaussian,
		},
		&MorphOpen{KernelSize: 2},
	)
}

// HighContrastStrategy returns a pipeline for very low contrast documents.
// Aggressive contrast enhancement for nearly invisible barcodes.
//
// Pipeline: Grayscale → CLAHE(4.0) → AdaptiveThreshold(Gaussian, 21, 4)
func HighContrastStrategy() Strategy {
	return NewCompositeStrategy("high_contrast",
		&Grayscale{},
		&CLAHE{ClipLimit: 4.0, TileSize: 8},
		&AdaptiveThreshold{
			BlockSize: 21,
			C:         4,
			Method:    gocv.AdaptiveThresholdGaussian,
		},
	)
}

// SauvolaStrategy returns a pipeline using Sauvola binarization.
// Best for historical documents and very uneven lighting.
//
// Pipeline: Grayscale → Median(3) → Sauvola(15, 0.3)
func SauvolaStrategy() Strategy {
	return NewCompositeStrategy("sauvola",
		&Grayscale{},
		&MedianBlur{KernelSize: 3},
		&SauvolaThreshold{
			WindowSize: 15,
			K:          0.3,
		},
	)
}

// WolfStrategy returns a pipeline using Wolf binarization.
// Alternative to Sauvola for documents with extreme backgrounds.
//
// Pipeline: Grayscale → Median(3) → Wolf(15, 0.5)
func WolfStrategy() Strategy {
	return NewCompositeStrategy("wolf",
		&Grayscale{},
		&MedianBlur{KernelSize: 3},
		&WolfThreshold{
			WindowSize: 15,
			K:          0.5,
		},
	)
}

// RotationStrategy returns a strategy that tries multiple rotation angles.
// Use when documents may be slightly skewed.
//
// Each rotation variant applies StandardStrategy preprocessing.
func RotationStrategy() Strategy {
	return &rotationWithPreprocess{
		base: &RotationSearch{
			Angles: []float64{-5, -3, -1, 1, 3, 5},
		},
		preprocess: StandardStrategy(),
	}
}

// DeskewStrategy returns a strategy that attempts automatic skew correction.
// Uses Hough line detection to find the dominant angle.
//
// Pipeline: Deskew → Standard preprocessing
func DeskewStrategy() Strategy {
	return NewCompositeStrategy("deskew",
		&Deskew{MaxAngle: 15},
		StandardStrategy(),
	)
}

// BlueChannelStrategy extracts the blue channel before processing.
// Sometimes effective for documents with colored backgrounds.
//
// Pipeline: BlueChannel → CLAHE → AdaptiveThreshold
func BlueChannelStrategy() Strategy {
	return NewCompositeStrategy("blue_channel",
		&BlueChannel{},
		&CLAHE{ClipLimit: 2.0, TileSize: 8},
		&AdaptiveThreshold{
			BlockSize: 11,
			C:         2,
			Method:    gocv.AdaptiveThresholdGaussian,
		},
	)
}

// NLMeansStrategy uses non-local means denoising.
// Very effective but slow. Use for critical documents.
//
// Pipeline: Grayscale → FastNlMeansDenoising(10) → CLAHE → AdaptiveThreshold
func NLMeansStrategy() Strategy {
	return NewCompositeStrategy("nlmeans",
		&Grayscale{},
		&FastNlMeansDenoising{H: 10},
		&CLAHE{ClipLimit: 2.0, TileSize: 8},
		&AdaptiveThreshold{
			BlockSize: 11,
			C:         2,
			Method:    gocv.AdaptiveThresholdGaussian,
		},
	)
}

// rotationWithPreprocess combines rotation variants with a preprocessing strategy.
type rotationWithPreprocess struct {
	base       *RotationSearch
	preprocess Strategy
}

func (r *rotationWithPreprocess) Name() string {
	return "rotation+" + r.preprocess.Name()
}

func (r *rotationWithPreprocess) Apply(src gocv.Mat) (gocv.Mat, error) {
	// Apply first rotation, then preprocess
	rotated, err := r.base.Apply(src)
	if err != nil {
		return gocv.Mat{}, err
	}
	defer rotated.Close()

	return r.preprocess.Apply(rotated)
}

// Variants implements VariantGenerator.
func (r *rotationWithPreprocess) Variants(src gocv.Mat) iter.Seq2[string, gocv.Mat] {
	return func(yield func(string, gocv.Mat) bool) {
		for variantName, rotated := range r.base.Variants(src) {
			processed, err := r.preprocess.Apply(rotated)
			rotated.Close()

			if err != nil {
				continue // Skip failed variants
			}

			if !yield(variantName, processed) {
				return
			}
		}
	}
}

// Ensure rotationWithPreprocess implements VariantGenerator.
var _ VariantGenerator = (*rotationWithPreprocess)(nil)

// QRDecodeStrategies returns the primary preprocessing strategies for QR code decoding.
// These cover the vast majority of real-world QR codes and are tried first:
//  1. identity          — original color image; let the decoder handle it as-is
//  2. inversion         — bitwise NOT for white-on-black QR codes
//  3. grayscale         — raw gray; lets ZBar use its own adaptive binarisation
//                         (critical for curved surfaces and uneven lighting)
//  4. gray_otsu         — grayscale → Otsu global threshold
//  5. sharpen_gray_blur3 — sharpen → grayscale → GaussianBlur(3)
//
// When all primary strategies fail, try QRDecodeFallbackStrategies.
func QRDecodeStrategies() []Strategy {
	return []Strategy{
		NewStrategyFunc("identity", func(src gocv.Mat) (gocv.Mat, error) {
			dst := gocv.NewMat()
			src.CopyTo(&dst)
			return dst, nil
		}),
		&Inversion{},
		&Grayscale{},
		NewCompositeStrategy("gray_otsu",
			&Grayscale{},
			&OtsuThreshold{},
		),
		NewCompositeStrategy("sharpen_gray_blur3",
			&Sharpen{},
			&Grayscale{},
			&GaussianBlur{KernelSize: 3},
		),
	}
}

// QRDecodeFallbackStrategies returns additional preprocessing strategies tried only
// when QRDecodeStrategies fails to produce a result. They cover less common cases
// that the primary set does not handle:
//  1. gray_blur5        — grayscale → GaussianBlur(5), no binarisation
//  2. gray_blur7        — grayscale → GaussianBlur(7), stronger blur
//  3. sharpen_gray      — sharpen → grayscale, raw (no binarisation)
//  4. sharpen_gray_otsu — sharpen → grayscale → Otsu
func QRDecodeFallbackStrategies() []Strategy {
	return []Strategy{
		NewCompositeStrategy("gray_blur5",
			&Grayscale{},
			&GaussianBlur{KernelSize: 5},
		),
		NewCompositeStrategy("gray_blur7",
			&Grayscale{},
			&GaussianBlur{KernelSize: 7},
		),
		NewCompositeStrategy("sharpen_gray",
			&Sharpen{},
			&Grayscale{},
		),
		NewCompositeStrategy("sharpen_gray_otsu",
			&Sharpen{},
			&Grayscale{},
			&OtsuThreshold{},
		),
	}
}
