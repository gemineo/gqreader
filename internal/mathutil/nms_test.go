package mathutil

import (
	"math"
	"testing"

	"github.com/mhey01/qreader/internal/geometry"
)

func TestNMS(t *testing.T) {
	tests := []struct {
		name       string
		boxes      []ScoredBox
		iouThresh  float32
		maxDets    int
		wantCount  int
		wantScores []float32
	}{
		{
			name: "no overlap - keep all",
			boxes: []ScoredBox{
				{Box: geometry.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, Score: 0.9, Index: 0},
				{Box: geometry.BBox{X1: 20, Y1: 20, X2: 30, Y2: 30}, Score: 0.8, Index: 1},
				{Box: geometry.BBox{X1: 40, Y1: 40, X2: 50, Y2: 50}, Score: 0.7, Index: 2},
			},
			iouThresh:  0.5,
			maxDets:    100,
			wantCount:  3,
			wantScores: []float32{0.9, 0.8, 0.7},
		},
		{
			name: "high overlap - suppress lower scores",
			boxes: []ScoredBox{
				{Box: geometry.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, Score: 0.9, Index: 0},
				{Box: geometry.BBox{X1: 1, Y1: 1, X2: 11, Y2: 11}, Score: 0.85, Index: 1}, // High overlap with first (IoU=0.68)
				{Box: geometry.BBox{X1: 2, Y1: 2, X2: 12, Y2: 12}, Score: 0.8, Index: 2},  // Medium overlap with first (IoU=0.47 < 0.5)
			},
			iouThresh:  0.5,
			maxDets:    100,
			wantCount:  2,                   // box0 and box2 kept, box1 suppressed
			wantScores: []float32{0.9, 0.8}, // Highest and third box kept
		},
		{
			name: "maxDets limit",
			boxes: []ScoredBox{
				{Box: geometry.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, Score: 0.9, Index: 0},
				{Box: geometry.BBox{X1: 20, Y1: 20, X2: 30, Y2: 30}, Score: 0.85, Index: 1},
				{Box: geometry.BBox{X1: 40, Y1: 40, X2: 50, Y2: 50}, Score: 0.8, Index: 2},
				{Box: geometry.BBox{X1: 60, Y1: 60, X2: 70, Y2: 70}, Score: 0.75, Index: 3},
			},
			iouThresh:  0.5,
			maxDets:    2,
			wantCount:  2,
			wantScores: []float32{0.9, 0.85}, // Top 2 by score
		},
		{
			name:       "empty input",
			boxes:      []ScoredBox{},
			iouThresh:  0.5,
			maxDets:    100,
			wantCount:  0,
			wantScores: []float32{},
		},
		{
			name: "single box",
			boxes: []ScoredBox{
				{Box: geometry.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, Score: 0.9, Index: 0},
			},
			iouThresh:  0.5,
			maxDets:    100,
			wantCount:  1,
			wantScores: []float32{0.9},
		},
		{
			name: "partial overlap - keep both",
			boxes: []ScoredBox{
				{Box: geometry.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, Score: 0.9, Index: 0},
				{Box: geometry.BBox{X1: 8, Y1: 8, X2: 18, Y2: 18}, Score: 0.8, Index: 1}, // Small overlap
			},
			iouThresh:  0.5,
			maxDets:    100,
			wantCount:  2,
			wantScores: []float32{0.9, 0.8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices := NMS(tt.boxes, tt.iouThresh, tt.maxDets)

			if len(indices) != tt.wantCount {
				t.Errorf("NMS() returned %d boxes, want %d", len(indices), tt.wantCount)
			}

			// Verify returned scores match expected (check by looking up scores)
			for i, idx := range indices {
				if i >= len(tt.wantScores) {
					break
				}
				// Find the box with this index
				var score float32
				for _, box := range tt.boxes {
					if box.Index == idx {
						score = box.Score
						break
					}
				}
				if score != tt.wantScores[i] {
					t.Errorf("NMS() result[%d] has score %v, want %v", i, score, tt.wantScores[i])
				}
			}
		})
	}
}

func TestSoftNMS(t *testing.T) {
	tests := []struct {
		name      string
		boxes     []ScoredBox
		iouThresh float32
		sigma     float32
		maxDets   int
		wantCount int
	}{
		{
			name: "basic soft NMS",
			boxes: []ScoredBox{
				{Box: geometry.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, Score: 0.9, Index: 0},
				{Box: geometry.BBox{X1: 1, Y1: 1, X2: 11, Y2: 11}, Score: 0.85, Index: 1}, // High overlap
			},
			iouThresh: 0.5,
			sigma:     0.5,
			maxDets:   100,
			wantCount: 2, // SoftNMS keeps both but reduces score of overlapping box
		},
		{
			name:      "empty input",
			boxes:     []ScoredBox{},
			iouThresh: 0.5,
			sigma:     0.5,
			maxDets:   100,
			wantCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices := SoftNMS(tt.boxes, tt.iouThresh, tt.sigma, 0.001, tt.maxDets)

			if len(indices) != tt.wantCount {
				t.Errorf("SoftNMS() returned %d boxes, want %d", len(indices), tt.wantCount)
			}
		})
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name     string
		input    float32
		expected float32
		epsilon  float32
	}{
		{"zero", 0, 0.5, 0.001},
		{"positive large", 10, 0.9999, 0.001},
		{"negative large", -10, 0.0001, 0.001},
		{"small positive", 1, 0.731, 0.01},
		{"small negative", -1, 0.269, 0.01},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Sigmoid(tt.input)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("Sigmoid(%v) = %v, want %v (±%v)", tt.input, result, tt.expected, tt.epsilon)
			}
		})
	}
}

func TestSigmoidSlice(t *testing.T) {
	input := []float32{-2, -1, 0, 1, 2}
	expected := []float32{
		Sigmoid(-2),
		Sigmoid(-1),
		Sigmoid(0),
		Sigmoid(1),
		Sigmoid(2),
	}

	// Make a copy since SigmoidSlice modifies in place
	testData := make([]float32, len(input))
	copy(testData, input)

	SigmoidSlice(testData)

	for i := range testData {
		if math.Abs(float64(testData[i]-expected[i])) > 0.001 {
			t.Errorf("SigmoidSlice()[%d] = %v, want %v", i, testData[i], expected[i])
		}
	}
}

func TestSigmoidSlice_Empty(t *testing.T) {
	empty := []float32{}
	SigmoidSlice(empty) // Should not panic
	if len(empty) != 0 {
		t.Errorf("SigmoidSlice([]) should not change empty slice length")
	}
}

func TestNMS_ZeroThreshold(t *testing.T) {
	boxes := []ScoredBox{
		{Box: geometry.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, Score: 0.9, Index: 0},
		{Box: geometry.BBox{X1: 1, Y1: 1, X2: 11, Y2: 11}, Score: 0.8, Index: 1},
	}

	// With IoU threshold of 0, any overlap should suppress
	result := NMS(boxes, 0.0, 100)

	// Should only keep the highest score
	if len(result) != 1 {
		t.Errorf("NMS() with threshold 0.0 returned %d boxes, want 1", len(result))
	}
}

func TestNMS_HighThreshold(t *testing.T) {
	boxes := []ScoredBox{
		{Box: geometry.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}, Score: 0.9, Index: 0},
		{Box: geometry.BBox{X1: 1, Y1: 1, X2: 11, Y2: 11}, Score: 0.8, Index: 1},
	}

	// With IoU threshold of 1.0, no suppression unless identical
	result := NMS(boxes, 1.0, 100)

	// Should keep both boxes
	if len(result) != 2 {
		t.Errorf("NMS() with threshold 1.0 returned %d boxes, want 2", len(result))
	}
}

// Benchmarks
func BenchmarkNMS_10Boxes(b *testing.B) {
	boxes := make([]ScoredBox, 10)
	for i := 0; i < 10; i++ {
		boxes[i] = ScoredBox{
			Box: geometry.BBox{
				X1: float32(i * 10),
				Y1: float32(i * 10),
				X2: float32(i*10 + 15),
				Y2: float32(i*10 + 15),
			},
			Score: 0.9 - float32(i)*0.05,
			Index: i,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NMS(boxes, 0.5, 100)
	}
}

func BenchmarkNMS_100Boxes(b *testing.B) {
	boxes := make([]ScoredBox, 100)
	for i := 0; i < 100; i++ {
		boxes[i] = ScoredBox{
			Box: geometry.BBox{
				X1: float32(i * 5),
				Y1: float32(i * 5),
				X2: float32(i*5 + 10),
				Y2: float32(i*5 + 10),
			},
			Score: 0.99 - float32(i)*0.001,
			Index: i,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NMS(boxes, 0.5, 100)
	}
}

func BenchmarkSoftNMS_10Boxes(b *testing.B) {
	boxes := make([]ScoredBox, 10)
	for i := 0; i < 10; i++ {
		boxes[i] = ScoredBox{
			Box: geometry.BBox{
				X1: float32(i * 10),
				Y1: float32(i * 10),
				X2: float32(i*10 + 15),
				Y2: float32(i*10 + 15),
			},
			Score: 0.9 - float32(i)*0.05,
			Index: i,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftNMS(boxes, 0.5, 0.5, 0.001, 100)
	}
}

func BenchmarkSigmoid(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sigmoid(1.5)
	}
}

func BenchmarkSigmoidSlice_1000(b *testing.B) {
	data := make([]float32, 1000)
	for i := range data {
		data[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		testData := make([]float32, len(data))
		copy(testData, data)
		SigmoidSlice(testData)
	}
}
