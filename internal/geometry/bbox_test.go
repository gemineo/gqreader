package geometry

import (
	"math"
	"testing"
)

func TestNewBBox(t *testing.T) {
	bbox := NewBBox(10, 20, 100, 200)
	if bbox.X1 != 10 || bbox.Y1 != 20 || bbox.X2 != 100 || bbox.Y2 != 200 {
		t.Errorf("NewBBox failed: got %v", bbox)
	}
}

func TestBBoxFromCenterWH(t *testing.T) {
	tests := []struct {
		name     string
		cx, cy   float32
		w, h     float32
		expected BBox
	}{
		{
			name: "centered box",
			cx:   50, cy: 50,
			w: 20, h: 30,
			expected: BBox{X1: 40, Y1: 35, X2: 60, Y2: 65},
		},
		{
			name: "origin box",
			cx:   10, cy: 10,
			w: 20, h: 20,
			expected: BBox{X1: 0, Y1: 0, X2: 20, Y2: 20},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bbox := FromCenterWH(tt.cx, tt.cy, tt.w, tt.h)
			if bbox != tt.expected {
				t.Errorf("FromCenterWH() = %v, want %v", bbox, tt.expected)
			}
		})
	}
}

func TestBBoxArea(t *testing.T) {
	tests := []struct {
		name     string
		bbox     BBox
		expected float32
	}{
		{
			name:     "standard box",
			bbox:     BBox{X1: 0, Y1: 0, X2: 10, Y2: 20},
			expected: 200,
		},
		{
			name:     "unit box",
			bbox:     BBox{X1: 0, Y1: 0, X2: 1, Y2: 1},
			expected: 1,
		},
		{
			name:     "offset box",
			bbox:     BBox{X1: 10, Y1: 10, X2: 20, Y2: 30},
			expected: 200,
		},
		{
			name:     "invalid box (reversed)",
			bbox:     BBox{X1: 20, Y1: 30, X2: 10, Y2: 10},
			expected: 200, // Implementation uses abs(), so reversed box still has area
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			area := tt.bbox.Area()
			if area != tt.expected {
				t.Errorf("Area() = %v, want %v", area, tt.expected)
			}
		})
	}
}

func TestBBoxCenter(t *testing.T) {
	tests := []struct {
		name      string
		bbox      BBox
		expectedX float32
		expectedY float32
	}{
		{
			name:      "centered at origin",
			bbox:      BBox{X1: -10, Y1: -10, X2: 10, Y2: 10},
			expectedX: 0,
			expectedY: 0,
		},
		{
			name:      "positive quadrant",
			bbox:      BBox{X1: 0, Y1: 0, X2: 100, Y2: 200},
			expectedX: 50,
			expectedY: 100,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cx, cy := tt.bbox.Center()
			if cx != tt.expectedX || cy != tt.expectedY {
				t.Errorf("Center() = (%v, %v), want (%v, %v)", cx, cy, tt.expectedX, tt.expectedY)
			}
		})
	}
}

func TestBBoxWidthHeight(t *testing.T) {
	tests := []struct {
		name      string
		bbox      BBox
		expectedW float32
		expectedH float32
	}{
		{
			name:      "square box",
			bbox:      BBox{X1: 0, Y1: 0, X2: 10, Y2: 10},
			expectedW: 10,
			expectedH: 10,
		},
		{
			name:      "rectangular box",
			bbox:      BBox{X1: 10, Y1: 20, X2: 30, Y2: 50},
			expectedW: 20,
			expectedH: 30,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, h := tt.bbox.WidthHeight()
			if w != tt.expectedW || h != tt.expectedH {
				t.Errorf("WidthHeight() = (%v, %v), want (%v, %v)", w, h, tt.expectedW, tt.expectedH)
			}
		})
	}
}

func TestBBoxIntersect(t *testing.T) {
	tests := []struct {
		name     string
		bbox1    BBox
		bbox2    BBox
		expected BBox
	}{
		{
			name:     "full overlap",
			bbox1:    BBox{X1: 0, Y1: 0, X2: 10, Y2: 10},
			bbox2:    BBox{X1: 5, Y1: 5, X2: 15, Y2: 15},
			expected: BBox{X1: 5, Y1: 5, X2: 10, Y2: 10},
		},
		{
			name:     "no overlap",
			bbox1:    BBox{X1: 0, Y1: 0, X2: 10, Y2: 10},
			bbox2:    BBox{X1: 20, Y1: 20, X2: 30, Y2: 30},
			expected: BBox{X1: 0, Y1: 0, X2: 0, Y2: 0}, // No overlap returns zero box
		},
		{
			name:     "identical boxes",
			bbox1:    BBox{X1: 10, Y1: 10, X2: 20, Y2: 20},
			bbox2:    BBox{X1: 10, Y1: 10, X2: 20, Y2: 20},
			expected: BBox{X1: 10, Y1: 10, X2: 20, Y2: 20},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox1.Intersect(tt.bbox2)
			if result != tt.expected {
				t.Errorf("Intersect() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBBoxUnion(t *testing.T) {
	tests := []struct {
		name     string
		bbox1    BBox
		bbox2    BBox
		expected BBox
	}{
		{
			name:     "adjacent boxes",
			bbox1:    BBox{X1: 0, Y1: 0, X2: 10, Y2: 10},
			bbox2:    BBox{X1: 10, Y1: 10, X2: 20, Y2: 20},
			expected: BBox{X1: 0, Y1: 0, X2: 20, Y2: 20},
		},
		{
			name:     "overlapping boxes",
			bbox1:    BBox{X1: 0, Y1: 0, X2: 15, Y2: 15},
			bbox2:    BBox{X1: 5, Y1: 5, X2: 20, Y2: 20},
			expected: BBox{X1: 0, Y1: 0, X2: 20, Y2: 20},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox1.Union(tt.bbox2)
			if result != tt.expected {
				t.Errorf("Union() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBBoxIoU(t *testing.T) {
	tests := []struct {
		name     string
		bbox1    BBox
		bbox2    BBox
		expected float32
		epsilon  float32
	}{
		{
			name:     "identical boxes",
			bbox1:    BBox{X1: 0, Y1: 0, X2: 10, Y2: 10},
			bbox2:    BBox{X1: 0, Y1: 0, X2: 10, Y2: 10},
			expected: 1.0,
			epsilon:  0.001,
		},
		{
			name:     "no overlap",
			bbox1:    BBox{X1: 0, Y1: 0, X2: 10, Y2: 10},
			bbox2:    BBox{X1: 20, Y1: 20, X2: 30, Y2: 30},
			expected: 0.0,
			epsilon:  0.001,
		},
		{
			name:     "half overlap",
			bbox1:    BBox{X1: 0, Y1: 0, X2: 10, Y2: 10},
			bbox2:    BBox{X1: 5, Y1: 0, X2: 15, Y2: 10},
			expected: 0.333, // intersection=50, union=150, iou=50/150=0.333
			epsilon:  0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox1.IoU(tt.bbox2)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("IoU() = %v, want %v (±%v)", result, tt.expected, tt.epsilon)
			}
		})
	}
}

func TestBBoxClip(t *testing.T) {
	tests := []struct {
		name     string
		bbox     BBox
		maxW     float32
		maxH     float32
		expected BBox
	}{
		{
			name:     "clip to bounds",
			bbox:     BBox{X1: -10, Y1: -10, X2: 110, Y2: 110},
			maxW:     100,
			maxH:     100,
			expected: BBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
		},
		{
			name:     "already within bounds",
			bbox:     BBox{X1: 10, Y1: 10, X2: 50, Y2: 50},
			maxW:     100,
			maxH:     100,
			expected: BBox{X1: 10, Y1: 10, X2: 50, Y2: 50},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox.Clip(tt.maxW, tt.maxH)
			if result != tt.expected {
				t.Errorf("Clip() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBBoxNormalize(t *testing.T) {
	tests := []struct {
		name     string
		bbox     BBox
		w        float32
		h        float32
		expected BBox
	}{
		{
			name:     "normalize to unit square",
			bbox:     BBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			w:        100,
			h:        100,
			expected: BBox{X1: 0, Y1: 0, X2: 1, Y2: 1},
		},
		{
			name:     "normalize half box",
			bbox:     BBox{X1: 0, Y1: 0, X2: 50, Y2: 100},
			w:        100,
			h:        200,
			expected: BBox{X1: 0, Y1: 0, X2: 0.5, Y2: 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox.Normalize(tt.w, tt.h)
			if result != tt.expected {
				t.Errorf("Normalize() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBBoxDenormalize(t *testing.T) {
	tests := []struct {
		name     string
		bbox     BBox
		w        float32
		h        float32
		expected BBox
	}{
		{
			name:     "denormalize from unit square",
			bbox:     BBox{X1: 0, Y1: 0, X2: 1, Y2: 1},
			w:        100,
			h:        100,
			expected: BBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
		},
		{
			name:     "denormalize partial box",
			bbox:     BBox{X1: 0.25, Y1: 0.25, X2: 0.75, Y2: 0.75},
			w:        200,
			h:        200,
			expected: BBox{X1: 50, Y1: 50, X2: 150, Y2: 150},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox.Denormalize(tt.w, tt.h)
			if result != tt.expected {
				t.Errorf("Denormalize() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBBoxScale(t *testing.T) {
	tests := []struct {
		name     string
		bbox     BBox
		factor   float32
		expected BBox
	}{
		{
			name:     "double size",
			bbox:     BBox{X1: 10, Y1: 10, X2: 20, Y2: 20},
			factor:   2.0,
			expected: BBox{X1: 5, Y1: 5, X2: 25, Y2: 25},
		},
		{
			name:     "half size",
			bbox:     BBox{X1: 10, Y1: 10, X2: 30, Y2: 30},
			factor:   0.5,
			expected: BBox{X1: 15, Y1: 15, X2: 25, Y2: 25},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox.Scale(tt.factor)
			if result != tt.expected {
				t.Errorf("Scale() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBBoxContains(t *testing.T) {
	bbox := BBox{X1: 10, Y1: 10, X2: 50, Y2: 50}

	tests := []struct {
		name     string
		x, y     float32
		expected bool
	}{
		{"inside", 30, 30, true},
		{"on edge", 10, 10, true},
		{"outside left", 5, 30, false},
		{"outside right", 55, 30, false},
		{"outside top", 30, 5, false},
		{"outside bottom", 30, 55, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := bbox.Contains(tt.x, tt.y)
			if result != tt.expected {
				t.Errorf("Contains(%v, %v) = %v, want %v", tt.x, tt.y, result, tt.expected)
			}
		})
	}
}

func TestBBoxIsValid(t *testing.T) {
	tests := []struct {
		name     string
		bbox     BBox
		expected bool
	}{
		{"valid box", BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, true},
		{"invalid - reversed X", BBox{X1: 10, Y1: 0, X2: 0, Y2: 10}, false},
		{"invalid - reversed Y", BBox{X1: 0, Y1: 10, X2: 10, Y2: 0}, false},
		{"invalid - both reversed", BBox{X1: 10, Y1: 10, X2: 0, Y2: 0}, false},
		{"zero size", BBox{X1: 5, Y1: 5, X2: 5, Y2: 5}, false}, // Zero area is invalid
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox.IsValid()
			if result != tt.expected {
				t.Errorf("IsValid() = %v, want %v for bbox %v", result, tt.expected, tt.bbox)
			}
		})
	}
}

func TestToCenterWH(t *testing.T) {
	tests := []struct {
		name       string
		bbox       BBox
		expectedCX float32
		expectedCY float32
		expectedW  float32
		expectedH  float32
	}{
		{
			name:       "centered box",
			bbox:       BBox{X1: 0, Y1: 0, X2: 10, Y2: 20},
			expectedCX: 5,
			expectedCY: 10,
			expectedW:  10,
			expectedH:  20,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cx, cy, w, h := tt.bbox.ToCenterWH()
			if cx != tt.expectedCX || cy != tt.expectedCY || w != tt.expectedW || h != tt.expectedH {
				t.Errorf("ToCenterWH() = (%v, %v, %v, %v), want (%v, %v, %v, %v)",
					cx, cy, w, h, tt.expectedCX, tt.expectedCY, tt.expectedW, tt.expectedH)
			}
		})
	}
}

func TestToXYXY(t *testing.T) {
	bbox := BBox{X1: 10, Y1: 20, X2: 30, Y2: 40}
	xyxy := bbox.ToXYXY()

	if xyxy[0] != 10 || xyxy[1] != 20 || xyxy[2] != 30 || xyxy[3] != 40 {
		t.Errorf("ToXYXY() = %v, want [10 20 30 40]", xyxy)
	}
}

// Benchmark tests
func BenchmarkBBoxArea(b *testing.B) {
	bbox := BBox{X1: 0, Y1: 0, X2: 100, Y2: 200}
	for i := 0; i < b.N; i++ {
		bbox.Area()
	}
}

func BenchmarkBBoxIoU(b *testing.B) {
	bbox1 := BBox{X1: 0, Y1: 0, X2: 100, Y2: 100}
	bbox2 := BBox{X1: 50, Y1: 50, X2: 150, Y2: 150}
	for i := 0; i < b.N; i++ {
		bbox1.IoU(bbox2)
	}
}

func BenchmarkBBoxIntersect(b *testing.B) {
	bbox1 := BBox{X1: 0, Y1: 0, X2: 100, Y2: 100}
	bbox2 := BBox{X1: 50, Y1: 50, X2: 150, Y2: 150}
	for i := 0; i < b.N; i++ {
		bbox1.Intersect(bbox2)
	}
}
