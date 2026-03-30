package qrdet

import (
	"math"
	"testing"
)

func TestDetection_Accessors(t *testing.T) {
	det := Detection{
		Confidence:    0.95,
		ImageWidth:    640,
		ImageHeight:   480,
		BBoxXYXY:      [4]float32{10, 20, 110, 120},
		BBoxXYXYN:     [4]float32{0.015625, 0.041667, 0.171875, 0.25},
		CenterXY:      [2]float32{60, 70},
		CenterXYN:     [2]float32{0.09375, 0.145833},
		WidthHeight:   [2]float32{100, 100},
		WidthHeightN:  [2]float32{0.15625, 0.208333},
		QuadXY:        [4][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}},
		QuadXYN:       [4][2]float32{{0.015625, 0.041667}, {0.171875, 0.041667}, {0.171875, 0.25}, {0.015625, 0.25}},
		PaddedQuadXY:  [4][2]float32{{5, 15}, {115, 15}, {115, 125}, {5, 125}},
		PaddedQuadXYN: [4][2]float32{{0.0078125, 0.03125}, {0.1796875, 0.03125}, {0.1796875, 0.260417}, {0.0078125, 0.260417}},
		PolygonXY:     [][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}},
		PolygonXYN:    [][2]float32{{0.015625, 0.041667}, {0.171875, 0.041667}, {0.171875, 0.25}, {0.015625, 0.25}},
	}

	if got := det.GetConfidence(); got != 0.95 {
		t.Errorf("GetConfidence() = %v, want 0.95", got)
	}

	if got := det.GetImageSize(); got != [2]int{640, 480} {
		t.Errorf("GetImageSize() = %v, want [640 480]", got)
	}

	if got := det.BBox(); got != det.BBoxXYXY {
		t.Errorf("BBox() = %v, want %v", got, det.BBoxXYXY)
	}

	if got := det.BBoxNormalized(); got != det.BBoxXYXYN {
		t.Errorf("BBoxNormalized() = %v, want %v", got, det.BBoxXYXYN)
	}

	if got := det.Center(); got != det.CenterXY {
		t.Errorf("Center() = %v, want %v", got, det.CenterXY)
	}

	if got := det.CenterNormalized(); got != det.CenterXYN {
		t.Errorf("CenterNormalized() = %v, want %v", got, det.CenterXYN)
	}

	if got := det.Size(); got != det.WidthHeight {
		t.Errorf("Size() = %v, want %v", got, det.WidthHeight)
	}

	if got := det.SizeNormalized(); got != det.WidthHeightN {
		t.Errorf("SizeNormalized() = %v, want %v", got, det.WidthHeightN)
	}

	if got := det.Quad(); got != det.QuadXY {
		t.Errorf("Quad() = %v, want %v", got, det.QuadXY)
	}

	if got := det.QuadNormalized(); got != det.QuadXYN {
		t.Errorf("QuadNormalized() = %v, want %v", got, det.QuadXYN)
	}

	if got := det.PaddedQuad(); got != det.PaddedQuadXY {
		t.Errorf("PaddedQuad() = %v, want %v", got, det.PaddedQuadXY)
	}

	if got := det.PaddedQuadNormalized(); got != det.PaddedQuadXYN {
		t.Errorf("PaddedQuadNormalized() = %v, want %v", got, det.PaddedQuadXYN)
	}

	if got := det.Polygon(); len(got) != 4 {
		t.Errorf("Polygon() length = %v, want 4", len(got))
	}

	if got := det.PolygonNormalized(); len(got) != 4 {
		t.Errorf("PolygonNormalized() length = %v, want 4", len(got))
	}
}

func TestDetection_Area(t *testing.T) {
	tests := []struct {
		name     string
		bbox     [4]float32
		wantArea float32
	}{
		{"square 100x100", [4]float32{10, 20, 110, 120}, 10000},
		{"rectangle 200x50", [4]float32{0, 0, 200, 50}, 10000},
		{"zero area", [4]float32{10, 10, 10, 10}, 0},
		{"small 1x1", [4]float32{0, 0, 1, 1}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			det := Detection{BBoxXYXY: tt.bbox}
			if got := det.Area(); got != tt.wantArea {
				t.Errorf("Area() = %v, want %v", got, tt.wantArea)
			}
		})
	}
}

func TestDetection_AspectRatio(t *testing.T) {
	tests := []struct {
		name      string
		bbox      [4]float32
		wantRatio float32
	}{
		{"square", [4]float32{0, 0, 100, 100}, 1.0},
		{"wide 2:1", [4]float32{0, 0, 200, 100}, 2.0},
		{"tall 1:2", [4]float32{0, 0, 100, 200}, 0.5},
		{"zero height", [4]float32{0, 0, 100, 0}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			det := Detection{BBoxXYXY: tt.bbox}
			got := det.AspectRatio()
			if math.Abs(float64(got-tt.wantRatio)) > 1e-5 {
				t.Errorf("AspectRatio() = %v, want %v", got, tt.wantRatio)
			}
		})
	}
}

func TestDetection_IsSquare(t *testing.T) {
	tests := []struct {
		name string
		bbox [4]float32
		want bool
	}{
		{"exact square", [4]float32{0, 0, 100, 100}, true},
		{"almost square 0.95", [4]float32{0, 0, 95, 100}, true},
		{"almost square 1.05", [4]float32{0, 0, 105, 100}, true},
		{"wide rectangle", [4]float32{0, 0, 200, 100}, false},
		{"tall rectangle", [4]float32{0, 0, 50, 100}, false},
		{"edge 0.9", [4]float32{0, 0, 90, 100}, true},
		{"edge 1.1", [4]float32{0, 0, 110, 100}, true},
		{"below 0.9", [4]float32{0, 0, 89, 100}, false},
		{"above 1.1", [4]float32{0, 0, 111, 100}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			det := Detection{BBoxXYXY: tt.bbox}
			if got := det.IsSquare(); got != tt.want {
				t.Errorf("IsSquare() = %v, want %v (ratio=%.3f)", got, tt.want, det.AspectRatio())
			}
		})
	}
}

func TestConfig_WithConfidence(t *testing.T) {
	orig := DefaultConfig()
	modified := orig.WithConfidence(0.8)

	if modified.ConfidenceThreshold != 0.8 {
		t.Errorf("WithConfidence() threshold = %v, want 0.8", modified.ConfidenceThreshold)
	}
	// Original must not change
	if orig.ConfidenceThreshold != 0.5 {
		t.Errorf("Original changed: ConfidenceThreshold = %v, want 0.5", orig.ConfidenceThreshold)
	}
}

func TestConfig_WithNMSIoU(t *testing.T) {
	orig := DefaultConfig()
	modified := orig.WithNMSIoU(0.6)

	if modified.NMSIoUThreshold != 0.6 {
		t.Errorf("WithNMSIoU() threshold = %v, want 0.6", modified.NMSIoUThreshold)
	}
	if orig.NMSIoUThreshold != 0.3 {
		t.Errorf("Original changed: NMSIoUThreshold = %v, want 0.3", orig.NMSIoUThreshold)
	}
}

func TestConfig_WithMaxDetections(t *testing.T) {
	orig := DefaultConfig()
	modified := orig.WithMaxDetections(50)

	if modified.MaxDetections != 50 {
		t.Errorf("WithMaxDetections() = %v, want 50", modified.MaxDetections)
	}
	if orig.MaxDetections != 100 {
		t.Errorf("Original changed: MaxDetections = %v, want 100", orig.MaxDetections)
	}
}

func TestConfig_Chaining(t *testing.T) {
	cfg := DefaultConfig().
		WithConfidence(0.7).
		WithNMSIoU(0.4).
		WithMaxDetections(25)

	if cfg.ConfidenceThreshold != 0.7 {
		t.Errorf("ConfidenceThreshold = %v, want 0.7", cfg.ConfidenceThreshold)
	}
	if cfg.NMSIoUThreshold != 0.4 {
		t.Errorf("NMSIoUThreshold = %v, want 0.4", cfg.NMSIoUThreshold)
	}
	if cfg.MaxDetections != 25 {
		t.Errorf("MaxDetections = %v, want 25", cfg.MaxDetections)
	}
	// InputSize should remain unchanged
	if cfg.InputSize != 640 {
		t.Errorf("InputSize = %v, want 640", cfg.InputSize)
	}
}

func TestDetection_NilPolygon(t *testing.T) {
	det := Detection{}
	if got := det.Polygon(); got != nil {
		t.Errorf("Polygon() on zero Detection = %v, want nil", got)
	}
	if got := det.PolygonNormalized(); got != nil {
		t.Errorf("PolygonNormalized() on zero Detection = %v, want nil", got)
	}
}
