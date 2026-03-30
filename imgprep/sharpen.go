package imgprep

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

// Sharpen applies an unsharp masking convolution to sharpen edges.
// Uses the kernel: [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
// which matches Python QReader's _SHARPEN_KERNEL.
type Sharpen struct {
	// Strength controls the sharpening intensity.
	// Values > 1.0 increase sharpening, values < 1.0 decrease it.
	// Zero or negative values default to 1.0.
	Strength float64
}

func (s *Sharpen) Name() string { return "sharpen" }

func (s *Sharpen) strength() float64 {
	if s.Strength <= 0 {
		return 1.0
	}
	return s.Strength
}

func (s *Sharpen) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	// Build sharpening kernel: center = 1 + 8*strength, neighbors = -strength
	// At strength=1.0 this gives [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
	str := s.strength()
	kernel := gocv.NewMatWithSize(3, 3, gocv.MatTypeCV32F)
	defer kernel.Close()

	neg := float32(-str)
	center := float32(1.0 + 8.0*str)
	kernel.SetFloatAt(0, 0, neg)
	kernel.SetFloatAt(0, 1, neg)
	kernel.SetFloatAt(0, 2, neg)
	kernel.SetFloatAt(1, 0, neg)
	kernel.SetFloatAt(1, 1, center)
	kernel.SetFloatAt(1, 2, neg)
	kernel.SetFloatAt(2, 0, neg)
	kernel.SetFloatAt(2, 1, neg)
	kernel.SetFloatAt(2, 2, neg)

	dst := gocv.NewMat()
	gocv.Filter2D(src, &dst, -1, kernel, image.Pt(-1, -1), 0, gocv.BorderDefault)
	return dst, nil
}
