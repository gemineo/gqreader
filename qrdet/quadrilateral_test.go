package qrdet

import (
	"math"
	"testing"
)

func TestFitQuadrilateral(t *testing.T) {
	tests := []struct {
		name    string
		polygon [][2]float32
	}{
		{
			name: "square polygon",
			polygon: [][2]float32{
				{0, 0}, {10, 0}, {10, 10}, {0, 10},
			},
		},
		{
			name: "triangle (less than 4 points)",
			polygon: [][2]float32{
				{0, 0}, {10, 0}, {5, 10},
			},
		},
		{
			name: "complex polygon (more than 4 points)",
			polygon: [][2]float32{
				{0, 0}, {5, 0}, {10, 0}, {10, 5}, {10, 10}, {5, 10}, {0, 10}, {0, 5},
			},
		},
		{
			name:    "empty polygon",
			polygon: [][2]float32{},
		},
		{
			name: "single point",
			polygon: [][2]float32{
				{5, 5},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fitted, expanded := fitQuadrilateral(tt.polygon)

			for _, result := range [][4][2]float32{fitted, expanded} {
				for i, point := range result {
					if math.IsNaN(float64(point[0])) || math.IsInf(float64(point[0]), 0) {
						t.Errorf("quad point[%d].x is invalid: %v", i, point[0])
					}
					if math.IsNaN(float64(point[1])) || math.IsInf(float64(point[1]), 0) {
						t.Errorf("quad point[%d].y is invalid: %v", i, point[1])
					}
				}
			}

			// For non-empty polygons, expanded quad bbox should contain the polygon
			if len(tt.polygon) >= 3 {
				minX, minY := tt.polygon[0][0], tt.polygon[0][1]
				maxX, maxY := tt.polygon[0][0], tt.polygon[0][1]
				for _, p := range tt.polygon {
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

				// Expanded quad bounding box should be at least as big as polygon bbox
				eMinX, eMinY := expanded[0][0], expanded[0][1]
				eMaxX, eMaxY := expanded[0][0], expanded[0][1]
				for _, p := range expanded {
					if p[0] < eMinX {
						eMinX = p[0]
					}
					if p[0] > eMaxX {
						eMaxX = p[0]
					}
					if p[1] < eMinY {
						eMinY = p[1]
					}
					if p[1] > eMaxY {
						eMaxY = p[1]
					}
				}

				tol := float32(2.0)
				if eMinX > minX+tol || eMaxX < maxX-tol || eMinY > minY+tol || eMaxY < maxY-tol {
					t.Errorf("expanded quad bbox [%.1f,%.1f]-[%.1f,%.1f] does not contain polygon bbox [%.1f,%.1f]-[%.1f,%.1f]",
						eMinX, eMinY, eMaxX, eMaxY, minX, minY, maxX, maxY)
				}
			}
		})
	}
}

func TestFitQuadrilateral_RealWorld(t *testing.T) {
	polygon := [][2]float32{
		{100, 100}, {105, 100}, {200, 105}, {205, 105},
		{205, 200}, {200, 205}, {105, 205}, {100, 200},
	}

	fitted, expanded := fitQuadrilateral(polygon)

	// Both should be valid
	for _, result := range [][4][2]float32{fitted, expanded} {
		for i, point := range result {
			if math.IsNaN(float64(point[0])) || math.IsInf(float64(point[0]), 0) {
				t.Errorf("quad point[%d].x is invalid: %v", i, point[0])
			}
			if math.IsNaN(float64(point[1])) || math.IsInf(float64(point[1]), 0) {
				t.Errorf("quad point[%d].y is invalid: %v", i, point[1])
			}
		}
	}

	// Fitted quad should have reasonable IoU with the polygon
	fittedArea := quadArea(fitted)
	if fittedArea < 1000 || fittedArea > 50000 {
		t.Errorf("fitted quad area = %v, expected reasonable area for ~100x100 polygon", fittedArea)
	}

	_ = expanded // expanded is tested via bbox containment in main test
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
			wantSize: 4,
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
			wantSize: -1,
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

			for i, point := range result {
				if math.IsNaN(float64(point[0])) || math.IsInf(float64(point[0]), 0) {
					t.Errorf("convexHull() point[%d].x is invalid: %v", i, point[0])
				}
				if math.IsNaN(float64(point[1])) || math.IsInf(float64(point[1]), 0) {
					t.Errorf("convexHull() point[%d].y is invalid: %v", i, point[1])
				}
			}

			if len(result) > len(tt.points) {
				t.Errorf("convexHull() returned more points (%d) than input (%d)", len(result), len(tt.points))
			}
		})
	}
}

func TestPolyArea(t *testing.T) {
	tests := []struct {
		name string
		pts  [][2]float64
		want float64
	}{
		{"unit square", [][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, 1.0},
		{"10x10 square", [][2]float64{{0, 0}, {10, 0}, {10, 10}, {0, 10}}, 100.0},
		{"triangle", [][2]float64{{0, 0}, {10, 0}, {5, 10}}, 50.0},
		{"degenerate (line)", [][2]float64{{0, 0}, {10, 10}}, 0},
		{"empty", nil, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := polyArea(tt.pts)
			if math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("polyArea() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPolyIoU(t *testing.T) {
	square := [][2]float64{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	squareArea := polyArea(square)

	tests := []struct {
		name    string
		p1, p2  [][2]float64
		p1Area  float64
		wantMin float64
		wantMax float64
	}{
		{
			name:    "identical squares",
			p1:      square,
			p2:      square,
			p1Area:  squareArea,
			wantMin: 0.99,
			wantMax: 1.01,
		},
		{
			name:    "non-overlapping",
			p1:      square,
			p2:      [][2]float64{{20, 20}, {30, 20}, {30, 30}, {20, 30}},
			p1Area:  squareArea,
			wantMin: 0,
			wantMax: 0.01,
		},
		{
			name:    "50% overlap",
			p1:      square,
			p2:      [][2]float64{{5, 0}, {15, 0}, {15, 10}, {5, 10}},
			p1Area:  squareArea,
			wantMin: 0.30,
			wantMax: 0.35,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := polyIoU(tt.p1, tt.p2, tt.p1Area)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("polyIoU() = %v, want in [%v, %v]", got, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestNelderMead3(t *testing.T) {
	// Minimize f(x,y,z) = (x-1)² + (y-2)² + (z-3)²
	fn := func(p [3]float64) float64 {
		return (p[0]-1)*(p[0]-1) + (p[1]-2)*(p[1]-2) + (p[2]-3)*(p[2]-3)
	}

	result := nelderMead3(fn, [3]float64{0, 0, 0}, 500)

	tol := 0.01
	if math.Abs(result[0]-1) > tol || math.Abs(result[1]-2) > tol || math.Abs(result[2]-3) > tol {
		t.Errorf("nelderMead3 = %v, want approximately [1, 2, 3]", result)
	}
}

// quadArea computes the area of a quadrilateral.
func quadArea(q [4][2]float32) float32 {
	pts := make([][2]float64, 4)
	for i, p := range q {
		pts[i] = [2]float64{float64(p[0]), float64(p[1])}
	}
	return float32(polyArea(pts))
}

// Benchmarks

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
