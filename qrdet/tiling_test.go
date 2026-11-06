package qrdet

import (
	"reflect"
	"testing"
)

func TestTileOrigins(t *testing.T) {
	tests := []struct {
		name    string
		n, size int
		overlap float32
		want    []int
	}{
		{"fits in one tile", 640, 1280, 0.2, []int{0}},
		{"exact tile", 1280, 1280, 0.2, []int{0}},
		{"2592 wide", 2592, 1280, 0.2, []int{0, 1024, 1312}},
		{"1920 high", 1920, 1280, 0.2, []int{0, 640}},
		{"no overlap", 3000, 1000, 0, []int{0, 1000, 2000}},
		{"zero size disables", 3000, 0, 0.2, []int{0}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tileOrigins(tc.n, tc.size, tc.overlap); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("tileOrigins(%d,%d,%v) = %v, want %v", tc.n, tc.size, tc.overlap, got, tc.want)
			}
		})
	}
}

func TestTouchesInnerEdge(t *testing.T) {
	tests := []struct {
		name string
		bbox [4]float32
		want bool
	}{
		{"inside", [4]float32{100, 100, 300, 300}, false},
		{"touches inner left", [4]float32{1, 100, 300, 300}, true},
		{"touches inner right", [4]float32{100, 100, 1279, 300}, true},
		{"touches bottom (image border, allowed)", [4]float32{100, 1000, 300, 1280}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Inner edges: left, top, right; bottom is the image border.
			if got := touchesInnerEdge(tc.bbox, 1280, 1280, true, true, true, false); got != tc.want {
				t.Errorf("touchesInnerEdge = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestMergeDetections(t *testing.T) {
	det := func(conf float32, box [4]float32) Detection {
		return Detection{Confidence: conf, BBoxXYXY: box}
	}
	dets := []Detection{
		det(0.6, [4]float32{0, 0, 100, 100}),
		det(0.9, [4]float32{5, 5, 105, 105}),     // duplicate of the first (IoU ~0.8)
		det(0.7, [4]float32{20, 20, 60, 60}),     // contained in the first
		det(0.8, [4]float32{500, 500, 600, 600}), // distinct
		det(0.5, [4]float32{590, 590, 700, 700}), // small overlap: kept
	}
	got := mergeDetections(dets, 0.3, 0.7)
	if len(got) != 3 {
		t.Fatalf("merged to %d detections, want 3", len(got))
	}
	if got[0].Confidence != 0.9 || got[1].Confidence != 0.8 || got[2].Confidence != 0.5 {
		t.Errorf("kept confidences = %v %v %v, want 0.9 0.8 0.5", got[0].Confidence, got[1].Confidence, got[2].Confidence)
	}
}

func TestOffsetDetection(t *testing.T) {
	src := buildDetection(0.9, 10, 20, 110, 120,
		[][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}},
		[4][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}},
		[4][2]float32{{8, 18}, {112, 18}, {112, 122}, {8, 122}},
		1280, 1280)
	got := offsetDetection(src, 1000, 500, 2592, 1920)
	if got.BBoxXYXY != [4]float32{1010, 520, 1110, 620} {
		t.Errorf("bbox = %v", got.BBoxXYXY)
	}
	if got.QuadXY[2] != [2]float32{1110, 620} || got.PaddedQuadXY[0] != [2]float32{1008, 518} {
		t.Errorf("quad = %v padded = %v", got.QuadXY, got.PaddedQuadXY)
	}
	if got.ImageWidth != 2592 || got.ImageHeight != 1920 {
		t.Errorf("image size = %dx%d", got.ImageWidth, got.ImageHeight)
	}
	if got.CenterXYN[0] < 0.40 || got.CenterXYN[0] > 0.41 {
		t.Errorf("normalised centre x = %v, want ~0.409", got.CenterXYN[0])
	}
	if got.Confidence != 0.9 || len(got.PolygonXY) != 4 || got.PolygonXY[0] != [2]float32{1010, 520} {
		t.Errorf("confidence/polygon not carried: %v %v", got.Confidence, got.PolygonXY)
	}
}
