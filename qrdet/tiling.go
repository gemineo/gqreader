package qrdet

import (
	"fmt"
	"image"
	"sort"

	"gocv.io/x/gocv"
)

// Tiled detection.
//
// The model sees a 640×640 letterboxed view of the image. On a 2592×1920
// photograph of a sheet of labels, a 90 px QR code shrinks to 22 px and is no
// longer detected. When tiling is enabled and the image is larger than one
// tile, the detector also runs on overlapping tiles, maps those detections
// back to image coordinates, discards the ones cut by an inner tile edge and
// merges everything with the full-image pass.

// tileOrigins returns the offsets of tiles of the given size along an axis of
// length n, with the requested overlap fraction. The last tile is aligned to
// the far edge so the whole axis is covered.
func tileOrigins(n, size int, overlap float32) []int {
	if size <= 0 || n <= size {
		return []int{0}
	}
	stride := int(float32(size) * (1 - overlap))
	if stride < 1 {
		stride = 1
	}
	var out []int
	for pos := 0; pos+size < n; pos += stride {
		out = append(out, pos)
	}
	last := n - size
	if len(out) == 0 || out[len(out)-1] != last {
		out = append(out, last)
	}
	return out
}

// detectTiles runs the detector on every tile and returns the detections in
// full-image coordinates.
func (d *Detector) detectTiles(mat gocv.Mat) ([]Detection, error) {
	size := d.config.TileSize
	fullW, fullH := mat.Cols(), mat.Rows()
	var out []Detection
	for _, oy := range tileOrigins(fullH, size, d.config.TileOverlap) {
		for _, ox := range tileOrigins(fullW, size, d.config.TileOverlap) {
			tw := min(size, fullW-ox)
			th := min(size, fullH-oy)
			roi := mat.Region(image.Rect(ox, oy, ox+tw, oy+th))
			dets, err := d.detectMat(roi)
			roi.Close()
			if err != nil {
				return nil, fmt.Errorf("tile (%d,%d): %w", ox, oy, err)
			}
			for _, det := range dets {
				if touchesInnerEdge(det.BBoxXYXY, tw, th, ox > 0, oy > 0, ox+tw < fullW, oy+th < fullH) {
					continue
				}
				out = append(out, offsetDetection(det, float32(ox), float32(oy), fullW, fullH))
			}
		}
	}
	return out, nil
}

// tileEdgeMargin is how close (in pixels) a bbox must come to a tile edge to
// count as cut by it.
const tileEdgeMargin = 2

// touchesInnerEdge reports whether a tile-space bbox reaches an edge of the
// tile that is not also an image border.
func touchesInnerEdge(bbox [4]float32, tw, th int, innerLeft, innerTop, innerRight, innerBottom bool) bool {
	m := float32(tileEdgeMargin)
	switch {
	case innerLeft && bbox[0] <= m:
		return true
	case innerTop && bbox[1] <= m:
		return true
	case innerRight && bbox[2] >= float32(tw)-m:
		return true
	case innerBottom && bbox[3] >= float32(th)-m:
		return true
	}
	return false
}

// offsetDetection translates a detection from tile to image coordinates and
// recomputes its normalised fields against the full image size.
func offsetDetection(det Detection, dx, dy float32, fullW, fullH int) Detection {
	polygon := make([][2]float32, len(det.PolygonXY))
	for i, p := range det.PolygonXY {
		polygon[i] = [2]float32{p[0] + dx, p[1] + dy}
	}
	var quad, padded [4][2]float32
	for i := 0; i < 4; i++ {
		quad[i] = [2]float32{det.QuadXY[i][0] + dx, det.QuadXY[i][1] + dy}
		padded[i] = [2]float32{det.PaddedQuadXY[i][0] + dx, det.PaddedQuadXY[i][1] + dy}
	}
	b := det.BBoxXYXY
	return buildDetection(det.Confidence, b[0]+dx, b[1]+dy, b[2]+dx, b[3]+dy, polygon, quad, padded, fullW, fullH)
}

// mergeDetections keeps, by decreasing confidence, every detection that does
// not overlap an already kept one: overlap means IoU above iouThreshold or
// one box mostly (containment above containThreshold) inside the other.
func mergeDetections(dets []Detection, iouThreshold, containThreshold float32) []Detection {
	sort.SliceStable(dets, func(i, j int) bool { return dets[i].Confidence > dets[j].Confidence })
	kept := make([]Detection, 0, len(dets))
	for _, d := range dets {
		dup := false
		for _, k := range kept {
			if boxOverlap(d.BBoxXYXY, k.BBoxXYXY, iouThreshold, containThreshold) {
				dup = true
				break
			}
		}
		if !dup {
			kept = append(kept, d)
		}
	}
	return kept
}

func boxOverlap(a, b [4]float32, iouThreshold, containThreshold float32) bool {
	ix := max(0, min(a[2], b[2])-max(a[0], b[0]))
	iy := max(0, min(a[3], b[3])-max(a[1], b[1]))
	inter := ix * iy
	if inter <= 0 {
		return false
	}
	areaA := (a[2] - a[0]) * (a[3] - a[1])
	areaB := (b[2] - b[0]) * (b[3] - b[1])
	union := areaA + areaB - inter
	if union > 0 && inter/union > iouThreshold {
		return true
	}
	smaller := min(areaA, areaB)
	return smaller > 0 && inter/smaller > containThreshold
}
