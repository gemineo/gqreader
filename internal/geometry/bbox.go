// Package geometry provides geometric primitives and operations for object detection.
package geometry

import "math"

// BBox represents an axis-aligned bounding box in [x1, y1, x2, y2] format.
// (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner.
type BBox struct {
	X1, Y1, X2, Y2 float32
}

// NewBBox creates a new bounding box from corner coordinates.
// Coordinates are automatically normalized so X1 <= X2 and Y1 <= Y2.
func NewBBox(x1, y1, x2, y2 float32) BBox {
	if x1 > x2 {
		x1, x2 = x2, x1
	}
	if y1 > y2 {
		y1, y2 = y2, y1
	}
	return BBox{X1: x1, Y1: y1, X2: x2, Y2: y2}
}

// FromCenterWH creates a bounding box from center coordinates and dimensions.
func FromCenterWH(cx, cy, w, h float32) BBox {
	halfW := w / 2
	halfH := h / 2
	return BBox{
		X1: cx - halfW,
		Y1: cy - halfH,
		X2: cx + halfW,
		Y2: cy + halfH,
	}
}

// Width returns the width of the bounding box.
func (b BBox) Width() float32 {
	return b.X2 - b.X1
}

// Height returns the height of the bounding box.
func (b BBox) Height() float32 {
	return b.Y2 - b.Y1
}

// Area returns the area of the bounding box.
func (b BBox) Area() float32 {
	return b.Width() * b.Height()
}

// Center returns the center point of the bounding box.
func (b BBox) Center() (cx, cy float32) {
	return (b.X1 + b.X2) / 2, (b.Y1 + b.Y2) / 2
}

// WidthHeight returns the width and height of the bounding box.
func (b BBox) WidthHeight() (w, h float32) {
	return b.Width(), b.Height()
}

// ToCenterWH converts the bounding box to center-width-height format.
func (b BBox) ToCenterWH() (cx, cy, w, h float32) {
	cx, cy = b.Center()
	w, h = b.WidthHeight()
	return
}

// ToXYXY returns the bounding box as [x1, y1, x2, y2] array.
func (b BBox) ToXYXY() [4]float32 {
	return [4]float32{b.X1, b.Y1, b.X2, b.Y2}
}

// Intersect returns the intersection of two bounding boxes.
// If the boxes don't intersect, returns a zero-area box.
func (b BBox) Intersect(other BBox) BBox {
	x1 := max32(b.X1, other.X1)
	y1 := max32(b.Y1, other.Y1)
	x2 := min32(b.X2, other.X2)
	y2 := min32(b.Y2, other.Y2)

	// If no intersection, return zero box
	if x1 >= x2 || y1 >= y2 {
		return BBox{}
	}

	return BBox{X1: x1, Y1: y1, X2: x2, Y2: y2}
}

// Union returns the smallest bounding box containing both boxes.
func (b BBox) Union(other BBox) BBox {
	return BBox{
		X1: min32(b.X1, other.X1),
		Y1: min32(b.Y1, other.Y1),
		X2: max32(b.X2, other.X2),
		Y2: max32(b.Y2, other.Y2),
	}
}

// IoU calculates the Intersection over Union between two bounding boxes.
// Returns a value between 0 (no overlap) and 1 (perfect overlap).
func (b BBox) IoU(other BBox) float32 {
	intersection := b.Intersect(other)
	intersectionArea := intersection.Area()

	if intersectionArea == 0 {
		return 0
	}

	unionArea := b.Area() + other.Area() - intersectionArea
	if unionArea == 0 {
		return 0
	}

	return intersectionArea / unionArea
}

// Clip constrains the bounding box to be within the given dimensions.
func (b BBox) Clip(width, height float32) BBox {
	return BBox{
		X1: clamp32(b.X1, 0, width),
		Y1: clamp32(b.Y1, 0, height),
		X2: clamp32(b.X2, 0, width),
		Y2: clamp32(b.Y2, 0, height),
	}
}

// Normalize converts pixel coordinates to normalized [0, 1] range.
func (b BBox) Normalize(width, height float32) BBox {
	return BBox{
		X1: b.X1 / width,
		Y1: b.Y1 / height,
		X2: b.X2 / width,
		Y2: b.Y2 / height,
	}
}

// Denormalize converts normalized coordinates back to pixel coordinates.
func (b BBox) Denormalize(width, height float32) BBox {
	return BBox{
		X1: b.X1 * width,
		Y1: b.Y1 * height,
		X2: b.X2 * width,
		Y2: b.Y2 * height,
	}
}

// Scale applies a scale factor to the bounding box dimensions around its center.
func (b BBox) Scale(factor float32) BBox {
	cx, cy := b.Center()
	w, h := b.WidthHeight()
	newW := w * factor
	newH := h * factor
	return FromCenterWH(cx, cy, newW, newH)
}

// Contains checks if a point is inside the bounding box.
func (b BBox) Contains(x, y float32) bool {
	return x >= b.X1 && x <= b.X2 && y >= b.Y1 && y <= b.Y2
}

// IsValid checks if the bounding box has positive area.
func (b BBox) IsValid() bool {
	return b.X2 > b.X1 && b.Y2 > b.Y1
}

// Helper functions

func min32(a, b float32) float32 {
	return float32(math.Min(float64(a), float64(b)))
}

func max32(a, b float32) float32 {
	return float32(math.Max(float64(a), float64(b)))
}

func clamp32(v, minVal, maxVal float32) float32 {
	if v < minVal {
		return minVal
	}
	if v > maxVal {
		return maxVal
	}
	return v
}
