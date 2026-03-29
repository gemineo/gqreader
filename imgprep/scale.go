package imgprep

import (
	"fmt"
	"image"
	"iter"

	"gocv.io/x/gocv"
)

// ScaleVariants generates multiple scaled versions of an image.
// Implements VariantGenerator to try different resolutions for QR decoding.
//
// Default factors match Python QReader: {0.5, 2, 0.25, 3, 4}.
// Scale factor 1.0 is excluded since the original image is tried separately.
type ScaleVariants struct {
	// Factors lists the scale factors to apply. Default: {0.5, 2, 0.25, 3, 4}.
	Factors []float64

	// MinSize is the minimum allowed dimension (width or height) after scaling.
	// Scales producing images smaller than this are skipped. Default: 25.
	MinSize int

	// MaxSize is the maximum allowed dimension after scaling.
	// Scales producing images larger than this are skipped. Default: 1024.
	MaxSize int
}

func (s *ScaleVariants) Name() string { return "scale" }

func (s *ScaleVariants) factors() []float64 {
	if len(s.Factors) == 0 {
		return []float64{0.5, 2, 0.25, 3, 4}
	}
	return s.Factors
}

func (s *ScaleVariants) minSize() int {
	if s.MinSize <= 0 {
		return 25
	}
	return s.MinSize
}

func (s *ScaleVariants) maxSize() int {
	if s.MaxSize <= 0 {
		return 1024
	}
	return s.MaxSize
}

// Apply returns the first valid scaled variant (for Strategy interface compatibility).
func (s *ScaleVariants) Apply(src gocv.Mat) (gocv.Mat, error) {
	for _, mat := range s.Variants(src) {
		return mat, nil
	}
	// No valid scale, return a copy
	dst := gocv.NewMat()
	src.CopyTo(&dst)
	return dst, nil
}

// Variants returns an iterator over scaled image variants.
// Each variant is named "scale_X.XX" where X.XX is the scale factor.
// Variants producing images outside [MinSize, MaxSize] are skipped.
func (s *ScaleVariants) Variants(src gocv.Mat) iter.Seq2[string, gocv.Mat] {
	return func(yield func(string, gocv.Mat) bool) {
		rows := src.Rows()
		cols := src.Cols()
		minDim := s.minSize()
		maxDim := s.maxSize()

		for _, factor := range s.factors() {
			newW := int(float64(cols) * factor)
			newH := int(float64(rows) * factor)

			// Skip if resulting size is out of bounds
			if newW < minDim || newH < minDim || newW > maxDim || newH > maxDim {
				continue
			}

			dst := gocv.NewMat()
			gocv.Resize(src, &dst, image.Pt(newW, newH), 0, 0, gocv.InterpolationCubic)

			name := fmt.Sprintf("scale_%.2f", factor)
			if !yield(name, dst) {
				return
			}
		}
	}
}

// Ensure ScaleVariants implements VariantGenerator.
var _ VariantGenerator = (*ScaleVariants)(nil)
