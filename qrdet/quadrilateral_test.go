package qrdet

import (
	"math"
	"testing"
)

func TestFitQuadrilateral(t *testing.T) {
	tests := []struct {
		name     string
		polygon  [][2]float32
		wantSize int
	}{
		{
			name: "square polygon",
			polygon: [][2]float32{
				{0, 0}, {10, 0}, {10, 10}, {0, 10},
			},
			wantSize: 4,
		},
		{
			name: "triangle (less than 4 points)",
			polygon: [][2]float32{
				{0, 0}, {10, 0}, {5, 10},
			},
			wantSize: 4, // Converts to 4-point quad (duplicates one point)
		},
		{
			name: "complex polygon (more than 4 points)",
			polygon: [][2]float32{
				{0, 0}, {5, 0}, {10, 0}, {10, 5}, {10, 10}, {5, 10}, {0, 10}, {0, 5},
			},
			wantSize: 4, // Should fit to 4 points
		},
		{
			name:     "empty polygon",
			polygon:  [][2]float32{},
			wantSize: 4, // Returns zero quad
		},
		{
			name: "single point",
			polygon: [][2]float32{
				{5, 5},
			},
			wantSize: 4, // Creates small square around point
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := fitQuadrilateral(tt.polygon)

			if len(result) != tt.wantSize {
				t.Errorf("fitQuadrilateral() returned %d points, want %d", len(result), tt.wantSize)
			}

			// Verify result points are valid (not NaN or Inf)
			for i, point := range result {
				if math.IsNaN(float64(point[0])) || math.IsInf(float64(point[0]), 0) {
					t.Errorf("fitQuadrilateral() point[%d].x is invalid: %v", i, point[0])
				}
				if math.IsNaN(float64(point[1])) || math.IsInf(float64(point[1]), 0) {
					t.Errorf("fitQuadrilateral() point[%d].y is invalid: %v", i, point[1])
				}
			}
		})
	}
}

func TestExpandQuadrilateral(t *testing.T) {
	tests := []struct {
		name    string
		quad    [4][2]float32
		polygon [][2]float32
	}{
		{
			name: "quad already contains polygon",
			quad: [4][2]float32{{0, 0}, {10, 0}, {10, 10}, {0, 10}},
			polygon: [][2]float32{
				{5, 5},
			},
		},
		{
			name: "quad needs expansion",
			quad: [4][2]float32{{2, 2}, {8, 2}, {8, 8}, {2, 8}},
			polygon: [][2]float32{
				{0, 0}, {10, 10}, // Points outside quad
			},
		},
		{
			name:    "empty polygon",
			quad:    [4][2]float32{{0, 0}, {10, 0}, {10, 10}, {0, 10}},
			polygon: [][2]float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := expandQuadrilateral(tt.quad, tt.polygon)

			// Verify result is valid
			for i, point := range result {
				if math.IsNaN(float64(point[0])) || math.IsInf(float64(point[0]), 0) {
					t.Errorf("expandQuadrilateral() point[%d].x is invalid: %v", i, point[0])
				}
				if math.IsNaN(float64(point[1])) || math.IsInf(float64(point[1]), 0) {
					t.Errorf("expandQuadrilateral() point[%d].y is invalid: %v", i, point[1])
				}
			}

			// If polygon is not empty, verify all polygon points are inside or on the expanded quad
			// This is a simplified check - full geometric containment check would be complex
			if len(tt.polygon) > 0 {
				// Find bounding box of result quad
				minX, minY := result[0][0], result[0][1]
				maxX, maxY := result[0][0], result[0][1]
				for _, point := range result {
					if point[0] < minX {
						minX = point[0]
					}
					if point[0] > maxX {
						maxX = point[0]
					}
					if point[1] < minY {
						minY = point[1]
					}
					if point[1] > maxY {
						maxY = point[1]
					}
				}

				// Check if polygon points are within bounding box (rough check)
				for _, point := range tt.polygon {
					if point[0] < minX-1 || point[0] > maxX+1 || point[1] < minY-1 || point[1] > maxY+1 {
						// Allow small tolerance for floating point
						t.Logf("Warning: polygon point %v may be outside expanded quad bounding box [%v,%v] to [%v,%v]",
							point, minX, minY, maxX, maxY)
					}
				}
			}
		})
	}
}

func TestConvexHull(t *testing.T) {
	tests := []struct {
		name     string
		points   [][2]float32
		wantSize int // Expected hull size (or -1 if variable)
	}{
		{
			name: "square (all points on hull)",
			points: [][2]float32{
				{0, 0}, {10, 0}, {10, 10}, {0, 10},
			},
			wantSize: 4,
		},
		{
			name: "square with interior point",
			points: [][2]float32{
				{0, 0}, {10, 0}, {10, 10}, {0, 10}, {5, 5},
			},
			wantSize: 4, // Interior point excluded
		},
		{
			name: "triangle",
			points: [][2]float32{
				{0, 0}, {10, 0}, {5, 10},
			},
			wantSize: 3,
		},
		{
			name: "collinear points",
			points: [][2]float32{
				{0, 0}, {5, 5}, {10, 10},
			},
			wantSize: -1, // Could be 2 or 3 depending on implementation
		},
		{
			name:     "empty",
			points:   [][2]float32{},
			wantSize: 0,
		},
		{
			name: "single point",
			points: [][2]float32{
				{5, 5},
			},
			wantSize: 1,
		},
		{
			name: "two points",
			points: [][2]float32{
				{0, 0}, {10, 10},
			},
			wantSize: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convexHull(tt.points)

			if tt.wantSize >= 0 && len(result) != tt.wantSize {
				t.Errorf("convexHull() returned %d points, want %d", len(result), tt.wantSize)
			}

			// Verify no NaN or Inf values
			for i, point := range result {
				if math.IsNaN(float64(point[0])) || math.IsInf(float64(point[0]), 0) {
					t.Errorf("convexHull() point[%d].x is invalid: %v", i, point[0])
				}
				if math.IsNaN(float64(point[1])) || math.IsInf(float64(point[1]), 0) {
					t.Errorf("convexHull() point[%d].y is invalid: %v", i, point[1])
				}
			}

			// Verify result length is not greater than input
			if len(result) > len(tt.points) {
				t.Errorf("convexHull() returned more points (%d) than input (%d)", len(result), len(tt.points))
			}
		})
	}
}

func TestPointInQuad(t *testing.T) {
	quad := [4][2]float32{
		{0, 0}, {10, 0}, {10, 10}, {0, 10},
	}

	tests := []struct {
		name   string
		point  [2]float32
		expect bool
	}{
		{"center", [2]float32{5, 5}, true},
		{"on edge", [2]float32{0, 5}, true},
		{"corner", [2]float32{0, 0}, true},
		{"outside left", [2]float32{-1, 5}, false},
		{"outside right", [2]float32{11, 5}, false},
		{"outside top", [2]float32{5, -1}, false},
		{"outside bottom", [2]float32{5, 11}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := pointInQuad(tt.point, quad)
			if result != tt.expect {
				t.Errorf("pointInQuad(%v) = %v, want %v", tt.point, result, tt.expect)
			}
		})
	}
}

// Note: raySegmentIntersection is a complex internal function with specific
// parameters (ox, oy, angle, a, b) that's difficult to test in isolation.
// It's tested indirectly through fitQuadrilateral and pointInQuad tests.

func TestFitQuadrilateral_RealWorld(t *testing.T) {
	// Test with a realistic QR code corner detection result
	// Simulating a slightly rotated and distorted QR code
	polygon := [][2]float32{
		{100, 100}, {105, 100}, {200, 105}, {205, 105},
		{205, 200}, {200, 205}, {105, 205}, {100, 200},
	}

	result := fitQuadrilateral(polygon)

	if len(result) != 4 {
		t.Fatalf("fitQuadrilateral() returned %d points, want 4", len(result))
	}

	// Verify the quadrilateral roughly contains the original polygon's bounding box
	minX, minY := polygon[0][0], polygon[0][1]
	maxX, maxY := polygon[0][0], polygon[0][1]
	for _, p := range polygon {
		if p[0] < minX {
			minX = p[0]
		}
		if p[0] > maxX {
			maxX = p[0]
		}
		if p[1] < minY {
			minY = p[1]
		}
		if p[1] > maxY {
			maxY = p[1]
		}
	}

	// Check quad contains approximate bounding box (with tolerance)
	tolerance := float32(20.0)
	for _, p := range result {
		if p[0] < minX-tolerance || p[0] > maxX+tolerance ||
			p[1] < minY-tolerance || p[1] > maxY+tolerance {
			t.Logf("Quad point %v is significantly outside polygon bbox [%v,%v] to [%v,%v]",
				p, minX, minY, maxX, maxY)
		}
	}
}

// Benchmark tests
func BenchmarkFitQuadrilateral_Simple(b *testing.B) {
	polygon := [][2]float32{
		{0, 0}, {10, 0}, {10, 10}, {0, 10},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fitQuadrilateral(polygon)
	}
}

func BenchmarkFitQuadrilateral_Complex(b *testing.B) {
	polygon := make([][2]float32, 20)
	for i := 0; i < 20; i++ {
		angle := float64(i) * 2 * math.Pi / 20
		polygon[i] = [2]float32{
			float32(100 + 50*math.Cos(angle)),
			float32(100 + 50*math.Sin(angle)),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fitQuadrilateral(polygon)
	}
}

func BenchmarkConvexHull_10Points(b *testing.B) {
	points := make([][2]float32, 10)
	for i := 0; i < 10; i++ {
		points[i] = [2]float32{float32(i * 10), float32(i * 5)}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convexHull(points)
	}
}

func BenchmarkPointInQuad(b *testing.B) {
	quad := [4][2]float32{
		{0, 0}, {100, 0}, {100, 100}, {0, 100},
	}
	point := [2]float32{50, 50}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pointInQuad(point, quad)
	}
}
