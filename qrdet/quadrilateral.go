package qrdet

import (
	"math"
	"sort"
)

// fitQuadrilateral converts a polygon to a 4-point quadrilateral.
// Uses a simplified approach:
// 1. Compute convex hull
// 2. Find the minimum area bounding quadrilateral
//
// The resulting quadrilateral is ordered clockwise starting from top-left.
func fitQuadrilateral(polygon [][2]float32) [4][2]float32 {
	if len(polygon) < 3 {
		return [4][2]float32{}
	}

	// Compute convex hull
	hull := convexHull(polygon)

	if len(hull) < 3 {
		return [4][2]float32{}
	}

	// If hull already has 4 or fewer points, use those
	if len(hull) <= 4 {
		return hullToQuad(hull)
	}

	// Find the minimum area bounding quadrilateral
	// Simplified approach: find the 4 extreme points
	quad := findExtremePoints(hull)

	// Order points clockwise from top-left
	return orderQuadClockwise(quad)
}

// expandQuadrilateral expands a quadrilateral to fully contain all polygon points.
// This creates a "padded" version that can be used for cropping with margin.
func expandQuadrilateral(quad [4][2]float32, polygon [][2]float32) [4][2]float32 {
	if len(polygon) == 0 {
		return quad
	}

	// Find the centroid of the quad
	var cx, cy float32
	for i := 0; i < 4; i++ {
		cx += quad[i][0]
		cy += quad[i][1]
	}
	cx /= 4
	cy /= 4

	// Check if all polygon points are inside the quad
	// If not, expand the quad
	maxExpansion := float32(1.0)

	for _, p := range polygon {
		if !pointInQuad(p, quad) {
			// Calculate the expansion needed
			dist := pointToQuadDistance(p, quad, cx, cy)
			expansion := dist * 1.1 // 10% margin
			if expansion > maxExpansion {
				maxExpansion = expansion
			}
		}
	}

	// Expand the quad from its center
	var expanded [4][2]float32
	for i := 0; i < 4; i++ {
		dx := quad[i][0] - cx
		dy := quad[i][1] - cy
		expanded[i][0] = cx + dx*maxExpansion
		expanded[i][1] = cy + dy*maxExpansion
	}

	return expanded
}

// convexHull computes the convex hull of a set of points using Graham scan.
func convexHull(points [][2]float32) [][2]float32 {
	if len(points) < 3 {
		return points
	}

	// Find the bottom-most point (or left-most in case of tie)
	minIdx := 0
	for i := 1; i < len(points); i++ {
		if points[i][1] < points[minIdx][1] ||
			(points[i][1] == points[minIdx][1] && points[i][0] < points[minIdx][0]) {
			minIdx = i
		}
	}

	// Swap to first position
	points[0], points[minIdx] = points[minIdx], points[0]
	pivot := points[0]

	// Sort by polar angle with respect to pivot
	sortedPoints := make([][2]float32, len(points)-1)
	copy(sortedPoints, points[1:])

	sort.Slice(sortedPoints, func(i, j int) bool {
		angle1 := math.Atan2(float64(sortedPoints[i][1]-pivot[1]), float64(sortedPoints[i][0]-pivot[0]))
		angle2 := math.Atan2(float64(sortedPoints[j][1]-pivot[1]), float64(sortedPoints[j][0]-pivot[0]))
		if angle1 != angle2 {
			return angle1 < angle2
		}
		// If same angle, closer point first
		dist1 := squaredDistance(pivot, sortedPoints[i])
		dist2 := squaredDistance(pivot, sortedPoints[j])
		return dist1 < dist2
	})

	// Build hull using stack
	hull := make([][2]float32, 0, len(points))
	hull = append(hull, pivot)

	for _, p := range sortedPoints {
		// Remove points that make clockwise turn
		for len(hull) > 1 && crossProduct(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}

	return hull
}

// crossProduct computes the cross product of vectors (o->a) and (o->b).
// Positive = counterclockwise, negative = clockwise, zero = collinear.
func crossProduct(o, a, b [2]float32) float32 {
	return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
}

// squaredDistance computes the squared distance between two points.
func squaredDistance(a, b [2]float32) float32 {
	dx := b[0] - a[0]
	dy := b[1] - a[1]
	return dx*dx + dy*dy
}

// hullToQuad converts a hull with <=4 points to a quadrilateral.
func hullToQuad(hull [][2]float32) [4][2]float32 {
	var quad [4][2]float32

	switch len(hull) {
	case 0:
		return quad
	case 1:
		// Single point: create a small square around it
		p := hull[0]
		quad[0] = [2]float32{p[0] - 1, p[1] - 1}
		quad[1] = [2]float32{p[0] + 1, p[1] - 1}
		quad[2] = [2]float32{p[0] + 1, p[1] + 1}
		quad[3] = [2]float32{p[0] - 1, p[1] + 1}
	case 2:
		// Line: create a thin rectangle
		p1, p2 := hull[0], hull[1]
		dx := p2[0] - p1[0]
		dy := p2[1] - p1[1]
		length := float32(math.Sqrt(float64(dx*dx + dy*dy)))
		if length < 0.001 {
			length = 1
		}
		nx := -dy / length // Normal vector
		ny := dx / length
		quad[0] = [2]float32{p1[0] + nx, p1[1] + ny}
		quad[1] = [2]float32{p2[0] + nx, p2[1] + ny}
		quad[2] = [2]float32{p2[0] - nx, p2[1] - ny}
		quad[3] = [2]float32{p1[0] - nx, p1[1] - ny}
	case 3:
		// Triangle: add a fourth point
		copy(quad[:3], hull)
		// Add midpoint of longest edge
		var maxDist float32
		var midpoint [2]float32
		for i := 0; i < 3; i++ {
			j := (i + 1) % 3
			dist := squaredDistance(hull[i], hull[j])
			if dist > maxDist {
				maxDist = dist
				midpoint = [2]float32{
					(hull[i][0] + hull[j][0]) / 2,
					(hull[i][1] + hull[j][1]) / 2,
				}
			}
		}
		quad[3] = midpoint
	case 4:
		copy(quad[:], hull)
	}

	return orderQuadClockwise(quad)
}

// findExtremePoints finds 4 extreme points from a convex hull.
// Uses the rotating calipers concept simplified.
func findExtremePoints(hull [][2]float32) [4][2]float32 {
	var quad [4][2]float32

	// Find extreme points in each direction
	minX, maxX := hull[0], hull[0]
	minY, maxY := hull[0], hull[0]

	for _, p := range hull[1:] {
		if p[0] < minX[0] {
			minX = p
		}
		if p[0] > maxX[0] {
			maxX = p
		}
		if p[1] < minY[1] {
			minY = p
		}
		if p[1] > maxY[1] {
			maxY = p
		}
	}

	quad[0] = minY // Top (minimum Y)
	quad[1] = maxX // Right
	quad[2] = maxY // Bottom (maximum Y)
	quad[3] = minX // Left

	return quad
}

// orderQuadClockwise orders quadrilateral points clockwise starting from top-left.
func orderQuadClockwise(quad [4][2]float32) [4][2]float32 {
	// Find centroid
	var cx, cy float32
	for i := 0; i < 4; i++ {
		cx += quad[i][0]
		cy += quad[i][1]
	}
	cx /= 4
	cy /= 4

	// Sort by angle from centroid
	points := quad[:]
	sort.Slice(points, func(i, j int) bool {
		angle1 := math.Atan2(float64(points[i][1]-cy), float64(points[i][0]-cx))
		angle2 := math.Atan2(float64(points[j][1]-cy), float64(points[j][0]-cx))
		return angle1 < angle2
	})

	// Find top-left (minimum x+y after sorting)
	topLeftIdx := 0
	minSum := points[0][0] + points[0][1]
	for i := 1; i < 4; i++ {
		sum := points[i][0] + points[i][1]
		if sum < minSum {
			minSum = sum
			topLeftIdx = i
		}
	}

	// Rotate so top-left is first
	var result [4][2]float32
	for i := 0; i < 4; i++ {
		result[i] = points[(topLeftIdx+i)%4]
	}

	return result
}

// pointInQuad checks if a point is inside a quadrilateral.
func pointInQuad(p [2]float32, quad [4][2]float32) bool {
	// Use winding number algorithm
	windingNumber := 0

	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		if quad[i][1] <= p[1] {
			if quad[j][1] > p[1] {
				if crossProduct(quad[i], quad[j], p) > 0 {
					windingNumber++
				}
			}
		} else {
			if quad[j][1] <= p[1] {
				if crossProduct(quad[i], quad[j], p) < 0 {
					windingNumber--
				}
			}
		}
	}

	return windingNumber != 0
}

// pointToQuadDistance calculates how much the quad needs to expand to contain a point.
func pointToQuadDistance(p [2]float32, quad [4][2]float32, cx, cy float32) float32 {
	// Calculate the distance from center to point, relative to distance from center to quad edge
	distToPoint := float32(math.Sqrt(float64((p[0]-cx)*(p[0]-cx) + (p[1]-cy)*(p[1]-cy))))

	if distToPoint < 0.001 {
		return 1.0
	}

	// Find the intersection of the ray from center through point with the quad
	angle := math.Atan2(float64(p[1]-cy), float64(p[0]-cx))

	// Find the edge that the ray intersects
	var minT float32 = float32(math.MaxFloat32)
	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		t := raySegmentIntersection(cx, cy, float32(angle), quad[i], quad[j])
		if t > 0 && t < minT {
			minT = t
		}
	}

	if minT == float32(math.MaxFloat32) {
		return 1.0
	}

	return distToPoint / minT
}

// raySegmentIntersection finds the parameter t where ray from (ox, oy) at angle intersects segment.
func raySegmentIntersection(ox, oy, angle float32, a, b [2]float32) float32 {
	dx := float32(math.Cos(float64(angle)))
	dy := float32(math.Sin(float64(angle)))

	// Ray: P = O + t * D
	// Segment: P = A + s * (B - A)

	segDx := b[0] - a[0]
	segDy := b[1] - a[1]

	denom := dx*segDy - dy*segDx
	if math.Abs(float64(denom)) < 1e-10 {
		return -1 // Parallel
	}

	t := ((a[0]-ox)*segDy - (a[1]-oy)*segDx) / denom
	s := ((a[0]-ox)*dy - (a[1]-oy)*dx) / denom

	if s >= 0 && s <= 1 && t > 0 {
		return t
	}

	return -1
}
