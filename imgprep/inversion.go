package imgprep

import (
	"fmt"

	"gocv.io/x/gocv"
)

// Inversion applies bitwise NOT to invert image colors.
// Maps white to black and vice versa (255 - pixel value).
// Useful for QR codes printed as white-on-black.
type Inversion struct{}

func (inv *Inversion) Name() string { return "inversion" }

func (inv *Inversion) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	dst := gocv.NewMat()
	gocv.BitwiseNot(src, &dst)
	return dst, nil
}
