package qrdet

// This file provides accessor methods for the Detection type.
//
// The Detection type currently exposes all fields publicly for backward compatibility
// (v0.x phase). In v1.0, these fields will be made private and only the methods below
// will remain in the public API.
//
// Users should migrate to using methods instead of direct field access:
//   - det.Confidence → det.GetConfidence()
//   - det.BBoxXYXY → det.BBox()
//   - det.BBoxXYXYN → det.BBoxNormalized()
//   - etc.

// GetConfidence returns the detection confidence score (0.0 to 1.0).
//
// Example:
//
//	for _, det := range detections {
//	    fmt.Printf("Confidence: %.2f%%\n", det.GetConfidence()*100)
//	}
func (d *Detection) GetConfidence() float32 {
	return d.Confidence
}

// GetImageSize returns the original image dimensions [width, height] in pixels.
func (d *Detection) GetImageSize() [2]int {
	return [2]int{d.ImageWidth, d.ImageHeight}
}

// BBox returns the bounding box in [x1, y1, x2, y2] format (pixels).
// (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner.
//
// Example:
//
//	bbox := det.BBox()
//	fmt.Printf("Top-left: (%.0f, %.0f), Bottom-right: (%.0f, %.0f)\n",
//	    bbox[0], bbox[1], bbox[2], bbox[3])
func (d *Detection) BBox() [4]float32 {
	return d.BBoxXYXY
}

// BBoxNormalized returns the bounding box normalized to [0, 1] range.
// Coordinates are relative to image dimensions.
//
// Example:
//
//	bboxN := det.BBoxNormalized()
//	// bboxN values are between 0.0 and 1.0
func (d *Detection) BBoxNormalized() [4]float32 {
	return d.BBoxXYXYN
}

// Center returns the center point of the bounding box (pixels).
//
// Example:
//
//	center := det.Center()
//	fmt.Printf("QR code center: (%.0f, %.0f)\n", center[0], center[1])
func (d *Detection) Center() [2]float32 {
	return d.CenterXY
}

// CenterNormalized returns the center point normalized to [0, 1] range.
func (d *Detection) CenterNormalized() [2]float32 {
	return d.CenterXYN
}

// Size returns the width and height of the bounding box (pixels).
//
// Example:
//
//	size := det.Size()
//	fmt.Printf("QR code size: %.0fx%.0f pixels\n", size[0], size[1])
func (d *Detection) Size() [2]float32 {
	return d.WidthHeight
}

// SizeNormalized returns the width and height normalized to [0, 1] range.
func (d *Detection) SizeNormalized() [2]float32 {
	return d.WidthHeightN
}

// Quad returns the fitted 4-point quadrilateral approximating the QR code boundary (pixels).
// Points are ordered clockwise starting from the top-left corner.
//
// This provides a better approximation of the QR code shape than the axis-aligned bounding box,
// especially for rotated QR codes.
//
// Example:
//
//	quad := det.Quad()
//	for i, pt := range quad {
//	    fmt.Printf("Corner %d: (%.0f, %.0f)\n", i+1, pt[0], pt[1])
//	}
func (d *Detection) Quad() [4][2]float32 {
	return d.QuadXY
}

// QuadNormalized returns the quadrilateral with coordinates normalized to [0, 1] range.
func (d *Detection) QuadNormalized() [4][2]float32 {
	return d.QuadXYN
}

// PaddedQuad returns an expanded quadrilateral that fully contains the segmentation polygon.
// This is useful for cropping the QR code region with some margin.
//
// The padding ensures that no part of the QR code is cut off when cropping.
//
// Example:
//
//	paddedQuad := det.PaddedQuad()
//	// Use paddedQuad for safe cropping with margin
func (d *Detection) PaddedQuad() [4][2]float32 {
	return d.PaddedQuadXY
}

// PaddedQuadNormalized returns the padded quadrilateral with coordinates normalized to [0, 1] range.
func (d *Detection) PaddedQuadNormalized() [4][2]float32 {
	return d.PaddedQuadXYN
}

// Polygon returns the precise segmentation polygon as a list of (x, y) points (pixels).
// This provides the most accurate boundary of the detected QR code.
//
// The polygon may have many points (typically 20-100) depending on the QR code shape.
//
// Example:
//
//	polygon := det.Polygon()
//	fmt.Printf("Polygon has %d vertices\n", len(polygon))
//	for i, pt := range polygon {
//	    fmt.Printf("  Vertex %d: (%.0f, %.0f)\n", i, pt[0], pt[1])
//	}
func (d *Detection) Polygon() [][2]float32 {
	return d.PolygonXY
}

// PolygonNormalized returns the segmentation polygon with coordinates normalized to [0, 1] range.
func (d *Detection) PolygonNormalized() [][2]float32 {
	return d.PolygonXYN
}

// Area returns the bounding box area in square pixels.
// This is a computed property based on the bounding box dimensions.
//
// Example:
//
//	area := det.Area()
//	fmt.Printf("QR code area: %.0f px²\n", area)
func (d *Detection) Area() float32 {
	w := d.BBoxXYXY[2] - d.BBoxXYXY[0]
	h := d.BBoxXYXY[3] - d.BBoxXYXY[1]
	return w * h
}

// AspectRatio returns the width/height ratio of the bounding box.
// For a perfect square QR code, this should be close to 1.0.
//
// Example:
//
//	ratio := det.AspectRatio()
//	if ratio < 0.9 || ratio > 1.1 {
//	    fmt.Println("Warning: QR code may be distorted")
//	}
func (d *Detection) AspectRatio() float32 {
	w := d.BBoxXYXY[2] - d.BBoxXYXY[0]
	h := d.BBoxXYXY[3] - d.BBoxXYXY[1]
	if h == 0 {
		return 0
	}
	return w / h
}

// IsSquare returns true if the QR code bounding box is approximately square.
// A detection is considered square if the aspect ratio is between 0.9 and 1.1.
//
// Example:
//
//	if det.IsSquare() {
//	    fmt.Println("QR code has normal square shape")
//	} else {
//	    fmt.Println("Warning: QR code may be distorted or cropped")
//	}
func (d *Detection) IsSquare() bool {
	ratio := d.AspectRatio()
	return ratio >= 0.9 && ratio <= 1.1
}
