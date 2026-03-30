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

// QRDecodeStrategies returns preprocessing strategies optimized for QR code decoding.
// These match the flavor pipeline used by Python QReader's decode loop:
//   - identity (original image, no preprocessing)
//   - inversion (bitwise NOT)
//   - grayscale → Otsu → GaussianBlur(5)
//   - sharpen → grayscale → GaussianBlur(3)
func QRDecodeStrategies() []Strategy {
	return []Strategy{
		NewStrategyFunc("identity", func(src gocv.Mat) (gocv.Mat, error) {
			dst := gocv.NewMat()
			src.CopyTo(&dst)
			return dst, nil
		}),
		&Inversion{},
		NewCompositeStrategy("gray_otsu_blur",
			&Grayscale{},
			&OtsuThreshold{},
			&GaussianBlur{KernelSize: 5},
		),
		NewCompositeStrategy("sharpen_gray_blur",
			&Sharpen{},
			&Grayscale{},
			&GaussianBlur{KernelSize: 3},
		),
	}
}
