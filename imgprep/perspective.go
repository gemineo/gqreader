package imgprep

import (
	"fmt"
	"image"
	"math"

	"gocv.io/x/gocv"
)

// PerspectiveCorrection applies a perspective warp to rectify a quadrilateral
// region into a square image. This is used to correct perspective distortion
// on detected QR codes before decoding.
//
// The Quad field must be set before calling Apply(). The quad corners are
// mapped to a square of side length max(width, height) of the quad.
type PerspectiveCorrection struct {
	// Quad defines the four source corner points (clockwise from top-left).
	Quad [4][2]float32
}

func (p *PerspectiveCorrection) Name() string { return "perspective" }

func (p *PerspectiveCorrection) Apply(src gocv.Mat) (gocv.Mat, error) {
	if src.Empty() {
		return gocv.Mat{}, fmt.Errorf("input image is empty")
	}

	// Compute output size from quad dimensions
	side := p.outputSize()
	if side < 1 {
		dst := gocv.NewMat()
		src.CopyTo(&dst)
		return dst, nil
	}

	// Source points (quad corners)
	srcPts := gocv.NewPoint2fVectorFromPoints([]gocv.Point2f{
		{X: p.Quad[0][0], Y: p.Quad[0][1]},
		{X: p.Quad[1][0], Y: p.Quad[1][1]},
		{X: p.Quad[2][0], Y: p.Quad[2][1]},
		{X: p.Quad[3][0], Y: p.Quad[3][1]},
	})
	defer srcPts.Close()

	// Destination points (a square)
	s := float32(side)
	dstPts := gocv.NewPoint2fVectorFromPoints([]gocv.Point2f{
		{X: 0, Y: 0},
		{X: s, Y: 0},
		{X: s, Y: s},
		{X: 0, Y: s},
	})
	defer dstPts.Close()

	transform := gocv.GetPerspectiveTransform2f(srcPts, dstPts)
	defer transform.Close()

	dst := gocv.NewMat()
	gocv.WarpPerspective(src, &dst, transform, image.Pt(side, side))
	return dst, nil
}

// outputSize computes the side length for the rectified square output.
func (p *PerspectiveCorrection) outputSize() int {
	topW := pointDist(p.Quad[0], p.Quad[1])
	botW := pointDist(p.Quad[3], p.Quad[2])
	leftH := pointDist(p.Quad[0], p.Quad[3])
	rightH := pointDist(p.Quad[1], p.Quad[2])

	maxW := math.Max(float64(topW), float64(botW))
	maxH := math.Max(float64(leftH), float64(rightH))

	return int(math.Max(maxW, maxH))
}

// pointDist computes the Euclidean distance between two 2D points.
func pointDist(a, b [2]float32) float64 {
	dx := float64(b[0] - a[0])
	dy := float64(b[1] - a[1])
	return math.Sqrt(dx*dx + dy*dy)
}
