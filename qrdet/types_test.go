package qrdet

import (
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.ConfidenceThreshold != 0.5 {
		t.Errorf("DefaultConfig().ConfidenceThreshold = %v, want 0.5", cfg.ConfidenceThreshold)
	}

	if cfg.NMSIoUThreshold != 0.3 {
		t.Errorf("DefaultConfig().NMSIoUThreshold = %v, want 0.3", cfg.NMSIoUThreshold)
	}

	if cfg.MaxDetections != 100 {
		t.Errorf("DefaultConfig().MaxDetections = %v, want 100", cfg.MaxDetections)
	}

	if cfg.InputSize != 640 {
		t.Errorf("DefaultConfig().InputSize = %v, want 640", cfg.InputSize)
	}
}

func TestConfig_Validation(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		valid  bool
	}{
		{
			name:   "default config is valid",
			config: DefaultConfig(),
			valid:  true,
		},
		{
			name: "high confidence threshold",
			config: Config{
				ConfidenceThreshold: 0.9,
				NMSIoUThreshold:     0.3,
				MaxDetections:       100,
				InputSize:           640,
			},
			valid: true,
		},
		{
			name: "low confidence threshold",
			config: Config{
				ConfidenceThreshold: 0.1,
				NMSIoUThreshold:     0.3,
				MaxDetections:       100,
				InputSize:           640,
			},
			valid: true,
		},
		{
			name: "edge case - zero confidence",
			config: Config{
				ConfidenceThreshold: 0.0,
				NMSIoUThreshold:     0.3,
				MaxDetections:       100,
				InputSize:           640,
			},
			valid: true,
		},
		{
			name: "edge case - max confidence",
			config: Config{
				ConfidenceThreshold: 1.0,
				NMSIoUThreshold:     0.3,
				MaxDetections:       100,
				InputSize:           640,
			},
			valid: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that config values are within expected ranges
			if tt.valid {
				if tt.config.ConfidenceThreshold < 0 || tt.config.ConfidenceThreshold > 1 {
					t.Errorf("ConfidenceThreshold out of range: %v", tt.config.ConfidenceThreshold)
				}
				if tt.config.NMSIoUThreshold < 0 || tt.config.NMSIoUThreshold > 1 {
					t.Errorf("NMSIoUThreshold out of range: %v", tt.config.NMSIoUThreshold)
				}
				if tt.config.MaxDetections <= 0 {
					t.Errorf("MaxDetections must be positive: %v", tt.config.MaxDetections)
				}
				if tt.config.InputSize <= 0 {
					t.Errorf("InputSize must be positive: %v", tt.config.InputSize)
				}
			}
		})
	}
}

func TestDetection_CoordinateAccess(t *testing.T) {
	det := Detection{
		Confidence:  0.95,
		BBoxXYXY:    [4]float32{10, 20, 110, 120},
		CenterXY:    [2]float32{60, 70},
		WidthHeight: [2]float32{100, 100},
		QuadXY:      [4][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}},
		PolygonXY:   [][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}},
	}

	// Test basic fields
	if det.Confidence != 0.95 {
		t.Errorf("Detection.Confidence = %v, want 0.95", det.Confidence)
	}

	// Test bounding box
	if det.BBoxXYXY[0] != 10 || det.BBoxXYXY[1] != 20 || det.BBoxXYXY[2] != 110 || det.BBoxXYXY[3] != 120 {
		t.Errorf("Detection.BBoxXYXY = %v, want [10 20 110 120]", det.BBoxXYXY)
	}

	// Test center
	if det.CenterXY[0] != 60 || det.CenterXY[1] != 70 {
		t.Errorf("Detection.CenterXY = %v, want [60 70]", det.CenterXY)
	}

	// Test width/height
	if det.WidthHeight[0] != 100 || det.WidthHeight[1] != 100 {
		t.Errorf("Detection.WidthHeight = %v, want [100 100]", det.WidthHeight)
	}

	// Test quadrilateral
	expectedQuad := [4][2]float32{{10, 20}, {110, 20}, {110, 120}, {10, 120}}
	if det.QuadXY != expectedQuad {
		t.Errorf("Detection.QuadXY = %v, want %v", det.QuadXY, expectedQuad)
	}

	// Test polygon length
	if len(det.PolygonXY) != 4 {
		t.Errorf("Detection.PolygonXY length = %v, want 4", len(det.PolygonXY))
	}
}

func TestDetection_NormalizedCoordinates(t *testing.T) {
	det := Detection{
		BBoxXYXYN:    [4]float32{0.1, 0.2, 0.9, 0.8},
		CenterXYN:    [2]float32{0.5, 0.5},
		WidthHeightN: [2]float32{0.8, 0.6},
		QuadXYN:      [4][2]float32{{0.1, 0.2}, {0.9, 0.2}, {0.9, 0.8}, {0.1, 0.8}},
		PolygonXYN:   [][2]float32{{0.1, 0.2}, {0.9, 0.2}, {0.9, 0.8}, {0.1, 0.8}},
	}

	// Verify all normalized coordinates are in [0, 1] range
	for i, v := range det.BBoxXYXYN {
		if v < 0 || v > 1 {
			t.Errorf("BBoxXYXYN[%d] = %v, outside [0,1] range", i, v)
		}
	}

	for i, v := range det.CenterXYN {
		if v < 0 || v > 1 {
			t.Errorf("CenterXYN[%d] = %v, outside [0,1] range", i, v)
		}
	}

	for i, v := range det.WidthHeightN {
		if v < 0 || v > 1 {
			t.Errorf("WidthHeightN[%d] = %v, outside [0,1] range", i, v)
		}
	}

	for i, point := range det.QuadXYN {
		for j, v := range point {
			if v < 0 || v > 1 {
				t.Errorf("QuadXYN[%d][%d] = %v, outside [0,1] range", i, j, v)
			}
		}
	}

	for i, point := range det.PolygonXYN {
		for j, v := range point {
			if v < 0 || v > 1 {
				t.Errorf("PolygonXYN[%d][%d] = %v, outside [0,1] range", i, j, v)
			}
		}
	}
}

func TestModelConstants(t *testing.T) {
	// Verify model format constants are correct for QRdet single-class model
	if modelOutputChannels != 37 {
		t.Errorf("modelOutputChannels = %v, want 37 for QRdet", modelOutputChannels)
	}

	if qrClassIndex != 4 {
		t.Errorf("qrClassIndex = %v, want 4 for single-class model", qrClassIndex)
	}

	if modelMaskCoefficients != 32 {
		t.Errorf("modelMaskCoefficients = %v, want 32", modelMaskCoefficients)
	}

	if modelPrototypeSize != 160 {
		t.Errorf("modelPrototypeSize = %v, want 160", modelPrototypeSize)
	}

	if modelOutputPredictions != 8400 {
		t.Errorf("modelOutputPredictions = %v, want 8400", modelOutputPredictions)
	}

	// Verify the relationship: 4 (bbox) + 1 (confidence) + 32 (mask coeffs) = 37
	expectedChannels := 4 + 1 + modelMaskCoefficients
	if modelOutputChannels != expectedChannels {
		t.Errorf("modelOutputChannels (%v) != 4 + 1 + maskCoeffs (%v)", modelOutputChannels, expectedChannels)
	}
}

func TestDetection_EmptyPolygon(t *testing.T) {
	det := Detection{
		Confidence: 0.8,
		PolygonXY:  [][2]float32{}, // Empty polygon
	}

	if det.PolygonXY == nil {
		t.Error("PolygonXY should not be nil, even when empty")
	}

	if len(det.PolygonXY) != 0 {
		t.Errorf("Empty PolygonXY length = %v, want 0", len(det.PolygonXY))
	}
}

func TestDetection_LargePolygon(t *testing.T) {
	// Create a detection with a large polygon (many vertices)
	polygonSize := 100
	polygon := make([][2]float32, polygonSize)
	for i := 0; i < polygonSize; i++ {
		polygon[i] = [2]float32{float32(i), float32(i)}
	}

	det := Detection{
		PolygonXY: polygon,
	}

	if len(det.PolygonXY) != polygonSize {
		t.Errorf("PolygonXY length = %v, want %v", len(det.PolygonXY), polygonSize)
	}
}

func TestConfig_CustomValues(t *testing.T) {
	cfg := Config{
		ConfidenceThreshold: 0.7,
		NMSIoUThreshold:     0.4,
		MaxDetections:       50,
		InputSize:           640,
	}

	if cfg.ConfidenceThreshold != 0.7 {
		t.Errorf("ConfidenceThreshold = %v, want 0.7", cfg.ConfidenceThreshold)
	}

	if cfg.NMSIoUThreshold != 0.4 {
		t.Errorf("NMSIoUThreshold = %v, want 0.4", cfg.NMSIoUThreshold)
	}

	if cfg.MaxDetections != 50 {
		t.Errorf("MaxDetections = %v, want 50", cfg.MaxDetections)
	}
}

func TestDetection_ZeroValues(t *testing.T) {
	// Test detection with zero values (edge case)
	det := Detection{
		Confidence:  0,
		BBoxXYXY:    [4]float32{0, 0, 0, 0},
		CenterXY:    [2]float32{0, 0},
		WidthHeight: [2]float32{0, 0},
	}

	if det.Confidence != 0 {
		t.Errorf("Zero confidence = %v, want 0", det.Confidence)
	}

	if det.WidthHeight[0] != 0 || det.WidthHeight[1] != 0 {
		t.Errorf("Zero width/height = %v, want [0 0]", det.WidthHeight)
	}
}

// Benchmark tests
func BenchmarkDefaultConfig(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = DefaultConfig()
	}
}

func BenchmarkDetection_Creation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = Detection{
			Confidence:  0.95,
			BBoxXYXY:    [4]float32{10, 20, 110, 120},
			CenterXY:    [2]float32{60, 70},
			WidthHeight: [2]float32{100, 100},
		}
	}
}
