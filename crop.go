package gqreader

import (
	"image"
	"math"

	"gocv.io/x/gocv"

	"github.com/gemineo/gqreader/qrdet"
)

// cropBBox extracts the axis-aligned bounding box region from the image.
// The returned Mat is a new allocation owned by the caller.
func cropBBox(img gocv.Mat, det qrdet.Detection) gocv.Mat {
	bbox := det.BBox()
	x1 := clampInt(int(bbox[0]), 0, img.Cols())
	y1 := clampInt(int(bbox[1]), 0, img.Rows())
	x2 := clampInt(int(bbox[2]), 0, img.Cols())
	y2 := clampInt(int(bbox[3]), 0, img.Rows())

	if x2 <= x1 || y2 <= y1 {
		// Degenerate bbox, return empty mat
		return gocv.NewMat()
	}

	roi := img.Region(image.Rect(x1, y1, x2, y2))
	defer roi.Close()

	dst := gocv.NewMat()
	roi.CopyTo(&dst)
	return dst
}

// cropPerspective extracts a perspective-corrected crop using the padded quad.
// Maps the quad corners to a square via WarpPerspective.
// The returned Mat is a new allocation owned by the caller.
func cropPerspective(img gocv.Mat, det qrdet.Detection) gocv.Mat {
	quad := det.PaddedQuad()

	// Compute output side length from quad dimensions
	side := perspectiveSide(quad)
	if side < 1 {
		return gocv.NewMat()
	}

	srcPts := gocv.NewPoint2fVectorFromPoints([]gocv.Point2f{
		{X: quad[0][0], Y: quad[0][1]},
		{X: quad[1][0], Y: quad[1][1]},
		{X: quad[2][0], Y: quad[2][1]},
		{X: quad[3][0], Y: quad[3][1]},
	})
	defer srcPts.Close()

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
	gocv.WarpPerspective(img, &dst, transform, image.Pt(side, side))
	return dst
}

// perspectiveSide computes the output side length for a perspective warp.
func perspectiveSide(quad [4][2]float32) int {
	topW := hypot(quad[1][0]-quad[0][0], quad[1][1]-quad[0][1])
	botW := hypot(quad[2][0]-quad[3][0], quad[2][1]-quad[3][1])
	leftH := hypot(quad[3][0]-quad[0][0], quad[3][1]-quad[0][1])
	rightH := hypot(quad[2][0]-quad[1][0], quad[2][1]-quad[1][1])

	maxW := topW
	if botW > maxW {
		maxW = botW
	}
	maxH := leftH
	if rightH > maxH {
		maxH = rightH
	}

	side := maxW
	if maxH > side {
		side = maxH
	}
	return int(side)
}

// hypot computes sqrt(dx² + dy²).
func hypot(dx, dy float32) float32 {
	return float32(math.Sqrt(float64(dx)*float64(dx) + float64(dy)*float64(dy)))
}

// clampInt clamps v to [lo, hi).
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v >= hi {
		return hi
	}
	return v
}
