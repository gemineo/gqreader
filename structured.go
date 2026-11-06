package gqreader

import (
	"image"

	"gocv.io/x/gocv"

	"github.com/gemineo/gqreader/qrdet"
	"github.com/gemineo/gqreader/qrstruct"
)

// structuredDecode is Phase 3 of the decode pipeline: it fits a sampling grid
// to the detection and decodes the module matrix directly (see package
// qrstruct). It only runs when every image-based strategy has failed, so it
// cannot regress detections that already decode.
func structuredDecode(img gocv.Mat, det qrdet.Detection, opts qrstruct.Options) (QRResult, bool) {
	result := QRResult{Detection: det}

	// Work on a region around the bbox rather than the whole image: the grid
	// never samples far outside the symbol, and converting a 4K frame to a Go
	// image for every failed detection would be wasteful.
	bbox := det.BBox()
	bw := float64(bbox[2] - bbox[0])
	bh := float64(bbox[3] - bbox[1])
	if bw < 4 || bh < 4 {
		return result, false
	}
	margin := 0.5 * max(bw, bh)
	x1 := clampInt(int(float64(bbox[0])-margin), 0, img.Cols())
	y1 := clampInt(int(float64(bbox[1])-margin), 0, img.Rows())
	x2 := clampInt(int(float64(bbox[2])+margin)+1, 0, img.Cols())
	y2 := clampInt(int(float64(bbox[3])+margin)+1, 0, img.Rows())
	if x2-x1 < 8 || y2-y1 < 8 {
		return result, false
	}

	gray, ok := grayRegion(img, image.Rect(x1, y1, x2, y2))
	if !ok {
		return result, false
	}

	// Try the fitted quad first (bbox corners first when the quad is
	// degenerate), then the other one: on symbols a few pixels wide the two
	// initialisations land in different basins and either may be the one
	// that decodes.
	quad, sane := saneQuad(det)
	inits := [][4][2]float32{quad}
	if box := bboxCorners(det); box != quad {
		if sane {
			inits = append(inits, box)
		} else {
			inits = append([][4][2]float32{box}, inits...)
		}
	}
	for _, init := range inits {
		var q qrstruct.Quad
		for i := range init {
			q[i] = qrstruct.Point{X: float64(init[i][0]) - float64(x1), Y: float64(init[i][1]) - float64(y1)}
		}
		r, err := qrstruct.Decode(gray, q, opts)
		if err != nil {
			continue
		}
		result.Content = r.Content
		result.Decoded = true
		result.Strategy = "structured/" + r.Stage
		return result, true
	}
	return result, false
}

// saneQuad returns the detection quad and whether all its corners lie within
// half a box side of the bbox; a degenerate quadrilateral fit fails that test.
func saneQuad(det qrdet.Detection) ([4][2]float32, bool) {
	quad := det.Quad()
	bbox := det.BBox()
	w := bbox[2] - bbox[0]
	h := bbox[3] - bbox[1]
	for _, p := range quad {
		if p[0] < bbox[0]-0.5*w || p[0] > bbox[2]+0.5*w || p[1] < bbox[1]-0.5*h || p[1] > bbox[3]+0.5*h {
			return quad, false
		}
	}
	return quad, true
}

// bboxCorners returns the axis-aligned bbox as a TL, TR, BR, BL quad.
func bboxCorners(det qrdet.Detection) [4][2]float32 {
	bbox := det.BBox()
	return [4][2]float32{
		{bbox[0], bbox[1]}, {bbox[2], bbox[1]}, {bbox[2], bbox[3]}, {bbox[0], bbox[3]},
	}
}

// grayRegion copies a rectangular region of img into a grayscale Go image.
func grayRegion(img gocv.Mat, rect image.Rectangle) (*image.Gray, bool) {
	roi := img.Region(rect)
	defer roi.Close()

	gray := gocv.NewMat()
	defer gray.Close()
	switch img.Channels() {
	case 1:
		roi.CopyTo(&gray)
	case 3:
		gocv.CvtColor(roi, &gray, gocv.ColorBGRToGray)
	case 4:
		gocv.CvtColor(roi, &gray, gocv.ColorBGRAToGray)
	default:
		return nil, false
	}
	if gray.Empty() || gray.Type() != gocv.MatTypeCV8U {
		return nil, false
	}

	out := image.NewGray(image.Rect(0, 0, gray.Cols(), gray.Rows()))
	copy(out.Pix, gray.ToBytes())
	return out, true
}
