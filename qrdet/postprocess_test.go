package qrdet

import (
	"math"
	"testing"

	"gocv.io/x/gocv"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name string
		x    float32
		want float32
		tol  float32
	}{
		{"zero", 0, 0.5, 1e-6},
		{"positive large", 10, 1.0, 1e-4},
		{"negative large", -10, 0.0, 1e-4},
		{"positive 1", 1, 0.7310586, 1e-5},
		{"negative 1", -1, 0.2689414, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sigmoid(tt.x)
			if math.Abs(float64(got-tt.want)) > float64(tt.tol) {
				t.Errorf("sigmoid(%v) = %v, want %v", tt.x, got, tt.want)
			}
		})
	}
}

func TestSigmoid_Symmetry(t *testing.T) {
	values := []float32{0.5, 1.0, 2.0, 5.0}
	for _, x := range values {
		sum := sigmoid(x) + sigmoid(-x)
		if math.Abs(float64(sum-1.0)) > 1e-5 {
			t.Errorf("sigmoid(%v) + sigmoid(%v) = %v, want 1.0", x, -x, sum)
		}
	}
}

func TestParseOutput0_EmptyOutput(t *testing.T) {
	// All zeros with high threshold → no detections
	output0 := make([]float32, modelOutputChannels*modelOutputPredictions)
	dets := parseOutput0(output0, 0.5)
	if len(dets) != 0 {
		t.Errorf("parseOutput0 with zeros returned %d detections, want 0", len(dets))
	}
}

func TestParseOutput0_SingleDetection(t *testing.T) {
	output0 := make([]float32, modelOutputChannels*modelOutputPredictions)

	// Place a detection at prediction index 0
	predIdx := 0
	output0[0*modelOutputPredictions+predIdx] = 320.0 // cx
	output0[1*modelOutputPredictions+predIdx] = 240.0 // cy
	output0[2*modelOutputPredictions+predIdx] = 100.0 // w
	output0[3*modelOutputPredictions+predIdx] = 100.0 // h
	output0[4*modelOutputPredictions+predIdx] = 0.9   // confidence

	// Set mask coefficients
	for j := 0; j < modelMaskCoefficients; j++ {
		output0[(5+j)*modelOutputPredictions+predIdx] = float32(j) * 0.1
	}

	dets := parseOutput0(output0, 0.5)
	if len(dets) != 1 {
		t.Fatalf("parseOutput0 returned %d detections, want 1", len(dets))
	}

	d := dets[0]
	if d.cx != 320.0 {
		t.Errorf("cx = %v, want 320", d.cx)
	}
	if d.cy != 240.0 {
		t.Errorf("cy = %v, want 240", d.cy)
	}
	if d.w != 100.0 {
		t.Errorf("w = %v, want 100", d.w)
	}
	if d.h != 100.0 {
		t.Errorf("h = %v, want 100", d.h)
	}
	if d.confidence != 0.9 {
		t.Errorf("confidence = %v, want 0.9", d.confidence)
	}
	if d.classIndex != 0 {
		t.Errorf("classIndex = %v, want 0", d.classIndex)
	}

	// Verify mask coefficients
	for j := 0; j < modelMaskCoefficients; j++ {
		expected := float32(j) * 0.1
		if math.Abs(float64(d.maskCoeffs[j]-expected)) > 1e-5 {
			t.Errorf("maskCoeffs[%d] = %v, want %v", j, d.maskCoeffs[j], expected)
		}
	}
}

func TestParseOutput0_MultipleDetections(t *testing.T) {
	output0 := make([]float32, modelOutputChannels*modelOutputPredictions)

	// Detection at index 0: confidence 0.9 (above threshold)
	output0[0*modelOutputPredictions+0] = 100
	output0[1*modelOutputPredictions+0] = 100
	output0[2*modelOutputPredictions+0] = 50
	output0[3*modelOutputPredictions+0] = 50
	output0[4*modelOutputPredictions+0] = 0.9

	// Detection at index 1: confidence 0.3 (below threshold of 0.5)
	output0[0*modelOutputPredictions+1] = 200
	output0[1*modelOutputPredictions+1] = 200
	output0[2*modelOutputPredictions+1] = 60
	output0[3*modelOutputPredictions+1] = 60
	output0[4*modelOutputPredictions+1] = 0.3

	// Detection at index 2: confidence 0.7 (above threshold)
	output0[0*modelOutputPredictions+2] = 300
	output0[1*modelOutputPredictions+2] = 300
	output0[2*modelOutputPredictions+2] = 80
	output0[3*modelOutputPredictions+2] = 80
	output0[4*modelOutputPredictions+2] = 0.7

	dets := parseOutput0(output0, 0.5)
	if len(dets) != 2 {
		t.Fatalf("parseOutput0 returned %d detections, want 2", len(dets))
	}

	// First returned detection should be index 0
	if dets[0].cx != 100 || dets[0].confidence != 0.9 {
		t.Errorf("dets[0] = {cx: %v, conf: %v}, want {cx: 100, conf: 0.9}", dets[0].cx, dets[0].confidence)
	}
	// Second should be index 2
	if dets[1].cx != 300 || dets[1].confidence != 0.7 {
		t.Errorf("dets[1] = {cx: %v, conf: %v}, want {cx: 300, conf: 0.7}", dets[1].cx, dets[1].confidence)
	}
}

func TestParseOutput0_ZeroThreshold(t *testing.T) {
	output0 := make([]float32, modelOutputChannels*modelOutputPredictions)

	// Place detections with very low confidence
	for i := 0; i < 3; i++ {
		output0[4*modelOutputPredictions+i] = 0.01
	}

	// With threshold 0.0, all should be returned (since 0.01 > 0.0)
	dets := parseOutput0(output0, 0.0)
	if len(dets) < 3 {
		t.Errorf("parseOutput0 with threshold 0.0 returned %d detections, want >= 3", len(dets))
	}
}

func TestBuildDetection(t *testing.T) {
	polygon := [][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}}
	quad := [4][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}}
	paddedQuad := [4][2]float32{{5, 15}, {115, 15}, {115, 125}, {5, 125}}

	det := buildDetection(0.95, 10, 20, 110, 120, polygon, quad, paddedQuad, 640, 480)

	// Verify pixel coordinates
	if det.Confidence != 0.95 {
		t.Errorf("Confidence = %v, want 0.95", det.Confidence)
	}
	if det.ImageWidth != 640 || det.ImageHeight != 480 {
		t.Errorf("ImageSize = (%d, %d), want (640, 480)", det.ImageWidth, det.ImageHeight)
	}
	if det.BBoxXYXY != [4]float32{10, 20, 110, 120} {
		t.Errorf("BBoxXYXY = %v, want [10 20 110 120]", det.BBoxXYXY)
	}

	// Verify computed center and size
	tol := float32(1e-3)
	if math.Abs(float64(det.CenterXY[0]-60)) > float64(tol) || math.Abs(float64(det.CenterXY[1]-70)) > float64(tol) {
		t.Errorf("CenterXY = %v, want [60 70]", det.CenterXY)
	}
	if math.Abs(float64(det.WidthHeight[0]-100)) > float64(tol) || math.Abs(float64(det.WidthHeight[1]-100)) > float64(tol) {
		t.Errorf("WidthHeight = %v, want [100 100]", det.WidthHeight)
	}

	// Verify normalized coordinates
	if math.Abs(float64(det.BBoxXYXYN[0]-10.0/640)) > float64(tol) {
		t.Errorf("BBoxXYXYN[0] = %v, want %v", det.BBoxXYXYN[0], 10.0/640)
	}
	if math.Abs(float64(det.BBoxXYXYN[1]-20.0/480)) > float64(tol) {
		t.Errorf("BBoxXYXYN[1] = %v, want %v", det.BBoxXYXYN[1], 20.0/480)
	}

	// Verify polygon normalization
	if len(det.PolygonXYN) != len(polygon) {
		t.Fatalf("PolygonXYN length = %d, want %d", len(det.PolygonXYN), len(polygon))
	}
	if math.Abs(float64(det.PolygonXYN[0][0]-10.0/640)) > float64(tol) {
		t.Errorf("PolygonXYN[0][0] = %v, want %v", det.PolygonXYN[0][0], 10.0/640)
	}

	// Verify quad normalization
	if math.Abs(float64(det.QuadXYN[0][0]-10.0/640)) > float64(tol) {
		t.Errorf("QuadXYN[0][0] = %v, want %v", det.QuadXYN[0][0], 10.0/640)
	}

	// Verify padded quad normalization
	if math.Abs(float64(det.PaddedQuadXYN[0][0]-5.0/640)) > float64(tol) {
		t.Errorf("PaddedQuadXYN[0][0] = %v, want %v", det.PaddedQuadXYN[0][0], 5.0/640)
	}
}

func TestBuildDetection_EmptyPolygon(t *testing.T) {
	det := buildDetection(0.5, 0, 0, 100, 100, nil, [4][2]float32{}, [4][2]float32{}, 640, 480)
	// buildDetection creates an empty (not nil) normalized polygon for nil input
	if len(det.PolygonXYN) != 0 {
		t.Errorf("Expected empty PolygonXYN for nil polygon input, got length %d", len(det.PolygonXYN))
	}
}

func BenchmarkParseOutput0(b *testing.B) {
	output0 := make([]float32, modelOutputChannels*modelOutputPredictions)
	// Place some detections
	for i := 0; i < 20; i++ {
		output0[4*modelOutputPredictions+i*400] = 0.8
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = parseOutput0(output0, 0.5)
	}
}

func TestClamp32f(t *testing.T) {
	tests := []struct {
		v, min, max, want float32
	}{
		{5, 0, 10, 5},
		{-1, 0, 10, 0},
		{15, 0, 10, 10},
		{0, 0, 10, 0},
		{10, 0, 10, 10},
	}

	for _, tt := range tests {
		got := clamp32f(tt.v, tt.min, tt.max)
		if got != tt.want {
			t.Errorf("clamp32f(%v, %v, %v) = %v, want %v", tt.v, tt.min, tt.max, got, tt.want)
		}
	}
}

func TestTransformCoordinates(t *testing.T) {
	info := &letterboxInfo{
		originalWidth:  800,
		originalHeight: 600,
		scale:          0.5,
		padLeft:        10,
		padTop:         20,
		targetSize:     640,
	}

	// (100, 100) in model space → remove pad → (90, 80) → /scale → (180, 160)
	x, y := info.transformCoordinates(100, 100)
	if math.Abs(float64(x-180)) > 1e-3 || math.Abs(float64(y-160)) > 1e-3 {
		t.Errorf("transformCoordinates(100,100) = (%v,%v), want (180,160)", x, y)
	}
}

func TestTransformCoordinates_Clamping(t *testing.T) {
	info := &letterboxInfo{
		originalWidth:  640,
		originalHeight: 480,
		scale:          1.0,
		padLeft:        0,
		padTop:         0,
		targetSize:     640,
	}

	// Negative coordinates should be clamped to 0
	x, y := info.transformCoordinates(-10, -20)
	if x != 0 || y != 0 {
		t.Errorf("transformCoordinates(-10,-20) = (%v,%v), want (0,0)", x, y)
	}

	// Coordinates beyond image should be clamped
	x, y = info.transformCoordinates(1000, 800)
	if x != 640 || y != 480 {
		t.Errorf("transformCoordinates(1000,800) = (%v,%v), want (640,480)", x, y)
	}
}

func TestTransformBBox(t *testing.T) {
	info := &letterboxInfo{
		originalWidth:  640,
		originalHeight: 480,
		scale:          1.0,
		padLeft:        0,
		padTop:         0,
		targetSize:     640,
	}

	x1, y1, x2, y2 := info.transformBBox(10, 20, 110, 120)
	if x1 != 10 || y1 != 20 || x2 != 110 || y2 != 120 {
		t.Errorf("transformBBox = (%v,%v,%v,%v), want (10,20,110,120)", x1, y1, x2, y2)
	}
}

func TestTransformPolygon(t *testing.T) {
	info := &letterboxInfo{
		originalWidth:  640,
		originalHeight: 480,
		scale:          1.0,
		padLeft:        0,
		padTop:         0,
		targetSize:     640,
	}

	polygon := [][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}}
	result := info.transformPolygon(polygon)

	if len(result) != 4 {
		t.Fatalf("transformPolygon length = %d, want 4", len(result))
	}
	if result[0] != [2]float32{10, 20} {
		t.Errorf("transformPolygon[0] = %v, want [10 20]", result[0])
	}
}

func TestModelStub(t *testing.T) {
	// model_stub.go provides these when not using embed_model tag
	data := GetEmbeddedModel()
	if data != nil {
		t.Errorf("GetEmbeddedModel() should return nil without embed_model tag")
	}
	if HasEmbeddedModel() {
		t.Error("HasEmbeddedModel() should return false without embed_model tag")
	}
	if EmbeddedModelSize() != 0 {
		t.Errorf("EmbeddedModelSize() = %d, want 0", EmbeddedModelSize())
	}
}

func TestMaskToPolygon_EmptyMask(t *testing.T) {
	emptyMask := gocv.NewMat()
	defer emptyMask.Close()

	// Empty mask → fallback to bbox rectangle
	result := maskToPolygon(emptyMask, 320, 240, 100, 100, 640)
	if len(result) != 4 {
		t.Fatalf("maskToPolygon(empty) length = %d, want 4 (bbox fallback)", len(result))
	}
	// Should form a rectangle centered at (320, 240) with w=100, h=100
	tol := float32(1e-3)
	if math.Abs(float64(result[0][0]-270)) > float64(tol) || math.Abs(float64(result[0][1]-190)) > float64(tol) {
		t.Errorf("maskToPolygon(empty)[0] = %v, want ~[270 190]", result[0])
	}
}

func TestMaskToPolygon_ValidMask(t *testing.T) {
	// Create a small binary mask with a filled region
	mask := gocv.NewMatWithSize(20, 20, gocv.MatTypeCV8UC1)
	defer mask.Close()

	// Fill a rectangle in the mask
	ptr, _ := mask.DataPtrUint8()
	for y := 5; y < 15; y++ {
		for x := 5; x < 15; x++ {
			ptr[y*20+x] = 255
		}
	}

	result := maskToPolygon(mask, 320, 240, 100, 100, 640)
	if len(result) < 3 {
		t.Errorf("maskToPolygon(valid) length = %d, want >= 3", len(result))
	}
}

func TestGenerateMask(t *testing.T) {
	// Create synthetic prototypes (32 channels, 160x160)
	protos := make([]float32, 32*160*160)

	// Set a pattern: all prototypes have value 1.0 in the center region
	for c := 0; c < 32; c++ {
		for y := 70; y < 90; y++ {
			for x := 70; x < 90; x++ {
				protos[c*160*160+y*160+x] = 1.0
			}
		}
	}

	// Coefficients that when multiplied should give large positive values
	coeffs := make([]float32, 32)
	for i := range coeffs {
		coeffs[i] = 0.5
	}

	mask := generateMask(coeffs, protos, 320, 320, 100, 100, 640)
	defer mask.Close()

	if mask.Empty() {
		t.Error("generateMask returned empty mask")
	}
}

func TestGenerateMask_ZeroArea(t *testing.T) {
	protos := make([]float32, 32*160*160)
	coeffs := make([]float32, 32)

	// Zero-width bbox → should return empty mask
	mask := generateMask(coeffs, protos, 320, 320, 0, 0, 640)
	defer mask.Close()

	if !mask.Empty() {
		t.Error("generateMask with zero-area bbox should return empty mask")
	}
}

// BenchmarkGenerateMask measures mask generation for a typical detection.
// This is the dominant CPU cost in postprocessing: a weighted sum of 32 prototype
// channels over every pixel in the bounding box, followed by a per-pixel threshold.
func BenchmarkGenerateMask(b *testing.B) {
	protos := make([]float32, modelMaskCoefficients*modelPrototypeSize*modelPrototypeSize)
	for i := range protos {
		protos[i] = float32(i%13-6) * 0.1 // synthetic but non-trivial values
	}
	var coeffs [modelMaskCoefficients]float32
	for i := range coeffs {
		coeffs[i] = float32(i%7-3) * 0.2
	}

	// Typical mid-size QR code: roughly 25% of the 640×640 input area
	cx, cy, w, h := float32(320), float32(320), float32(160), float32(160)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mask := generateMask(coeffs[:], protos, cx, cy, w, h, modelInputSize)
		mask.Close()
	}
}

// BenchmarkGenerateMask_Large measures mask generation for a large detection
// (QR code spanning most of the frame).
func BenchmarkGenerateMask_Large(b *testing.B) {
	protos := make([]float32, modelMaskCoefficients*modelPrototypeSize*modelPrototypeSize)
	for i := range protos {
		protos[i] = float32(i%17-8) * 0.05
	}
	var coeffs [modelMaskCoefficients]float32
	for i := range coeffs {
		coeffs[i] = float32(i%5-2) * 0.3
	}

	// Large: 75% of frame
	cx, cy, w, h := float32(320), float32(320), float32(480), float32(480)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mask := generateMask(coeffs[:], protos, cx, cy, w, h, modelInputSize)
		mask.Close()
	}
}

func BenchmarkBuildDetection(b *testing.B) {
	polygon := [][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}}
	quad := [4][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}}
	paddedQuad := [4][2]float32{{5, 15}, {115, 15}, {115, 125}, {5, 125}}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = buildDetection(0.95, 10, 20, 110, 120, polygon, quad, paddedQuad, 640, 480)
	}
}
