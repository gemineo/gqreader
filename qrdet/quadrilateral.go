package qrdet

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Fitting parameters matching Python's QuadrilateralFitter.fit() call in _qrdet_helpers.py:
//
//	QuadrilateralFitter(polygon).fit(
//	    simplify_polygons_larger_than=8,
//	    start_simplification_epsilon=0.1,
//	    max_simplification_epsilon=2.0,
//	    simplification_epsilon_increment=0.2,
//	)
const (
	fitterMaxHullSides    = 8
	fitterStartEpsilon    = 0.1
	fitterMaxEpsilon      = 2.0
	fitterEpsilonStep     = 0.2
	fitterIoUThreshold    = 0.8
	fitterMaxCombinations = 300
	fitterNMMaxIter       = 200

	// fitterSamplingSeed seeds the combination subsampling in findInitialQuad.
	// A fixed seed keeps detections reproducible across runs (required for
	// golden-file comparisons) while preserving the random-subset strategy.
	fitterSamplingSeed = 0x5152646574 // "QRdet"
)

// fitQuadrilateral fits a quadrilateral to a polygon using IoU-optimized search,
// matching Python's QuadrilateralFitter algorithm from the quadrilateral_fitter package.
//
// The algorithm:
//  1. Compute convex hull of the polygon
//  2. Simplify hull via Douglas-Peucker if it has too many vertices
//  3. Search all C(n,4) combinations for the quad with highest IoU against the hull
//  4. Refine edges via line fitting (Nelder-Mead minimization of perpendicular distances)
//  5. Expand fitted lines outward to contain all hull points
//
// Returns both the tight fitted quadrilateral and the expanded quadrilateral.
// Both are ordered clockwise in image coordinates (Y-axis down).
func fitQuadrilateral(polygon [][2]float32) (fitted, expanded [4][2]float32) {
	if len(polygon) < 3 {
		return
	}

	hull := convexHull(polygon)
	if len(hull) < 3 {
		return
	}

	poly64 := f32sToF64s(polygon)
	hull64 := f32sToF64s(hull)
	hullArea := polyArea(hull64)

	if len(hull64) <= 4 {
		q := hullToQuadF64(hull64)
		q = orderClockwiseF64(q)
		f := f64QuadToF32(q)
		return f, f
	}

	simplified := simplifyHullForFit(hull64, hullArea)
	initialQuad := findInitialQuad(simplified, hull64, hullArea)
	lines := finetuneLines(initialQuad, poly64)

	fittedF64 := linesToQuad(lines)
	fittedF64 = orderClockwiseF64(fittedF64)
	fitted = f64QuadToF32(fittedF64)

	expandedLines := expandLines(lines, hull64)
	expandedF64 := linesToQuad(expandedLines)
	expandedF64 = orderClockwiseF64(expandedF64)
	expanded = f64QuadToF32(expandedF64)

	return fitted, expanded
}

// findInitialQuad searches all 4-vertex combinations of the simplified hull
// for the quadrilateral with the highest IoU against the original hull.
func findInitialQuad(simplified, hull [][2]float64, hullArea float64) [4][2]float64 {
	n := len(simplified)
	if n < 4 {
		return hullToQuadF64(simplified)
	}

	type combo [4]int
	var combos []combo
	for i := 0; i < n-3; i++ {
		for j := i + 1; j < n-2; j++ {
			for k := j + 1; k < n-1; k++ {
				for l := k + 1; l < n; l++ {
					combos = append(combos, combo{i, j, k, l})
				}
			}
		}
	}

	if len(combos) > fitterMaxCombinations {
		rng := rand.New(rand.NewPCG(fitterSamplingSeed, uint64(len(combos))))
		for i := len(combos) - 1; i > 0; i-- {
			j := rng.IntN(i + 1)
			combos[i], combos[j] = combos[j], combos[i]
		}
		combos = combos[:fitterMaxCombinations]
	}

	bestIoU := 0.0
	var bestQuad [4][2]float64
	for _, c := range combos {
		quad := [][2]float64{simplified[c[0]], simplified[c[1]], simplified[c[2]], simplified[c[3]]}
		iou := polyIoU(hull, quad, hullArea)
		if iou > bestIoU {
			bestIoU = iou
			bestQuad = [4][2]float64{quad[0], quad[1], quad[2], quad[3]}
			if iou >= 1.0 {
				break
			}
		}
	}

	return bestQuad
}

// finetuneLines fits lines to the polygon points grouped by nearest quad edge.
// Each line is refined via Nelder-Mead minimization of the sum of absolute
// perpendicular distances, matching Python's scipy.optimize.minimize approach.
func finetuneLines(initialQuad [4][2]float64, polygon [][2]float64) [4]lineEq {
	var initial [4]lineEq
	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		initial[i] = lineFromPoints(
			initialQuad[i][0], initialQuad[i][1],
			initialQuad[j][0], initialQuad[j][1],
		)
	}

	// Assign each polygon point to its nearest line
	assignments := make([]int, len(polygon))
	for pi, pt := range polygon {
		minDist := math.MaxFloat64
		for li := 0; li < 4; li++ {
			d := initial[li].distance(pt[0], pt[1])
			if d < minDist {
				minDist = d
				assignments[pi] = li
			}
		}
	}

	var fitted [4]lineEq
	for li := 0; li < 4; li++ {
		var pts [][2]float64
		for pi, ai := range assignments {
			if ai == li {
				pts = append(pts, polygon[pi])
			}
		}

		if len(pts) < 2 {
			fitted[li] = initial[li]
			continue
		}

		// Objective: sum(|Ax+By+C|) / sqrt(A²+B²)  — matches Python's perpendicular_distance
		objective := func(p [3]float64) float64 {
			norm := math.Sqrt(p[0]*p[0] + p[1]*p[1])
			if norm < 1e-12 {
				return math.MaxFloat64
			}
			sum := 0.0
			for _, pt := range pts {
				sum += math.Abs(p[0]*pt[0] + p[1]*pt[1] + p[2])
			}
			return sum / norm
		}

		x0 := [3]float64{initial[li].A, initial[li].B, initial[li].C}
		result := nelderMead3(objective, x0, fitterNMMaxIter)
		fitted[li] = lineEq{A: result[0], B: result[1], C: result[2]}
	}

	return fitted
}

// expandLines moves each fitted line outward to contain all convex hull points.
// For each line, if any hull point is on the opposite side from the centroid,
// the line is shifted to pass through the farthest such point.
func expandLines(lines [4]lineEq, hull [][2]float64) [4]lineEq {
	var cx, cy float64
	for _, p := range hull {
		cx += p[0]
		cy += p[1]
	}
	cx /= float64(len(hull))
	cy /= float64(len(hull))

	var expanded [4]lineEq
	for i := 0; i < 4; i++ {
		expanded[i] = lines[i]
		centroidSign := signF64(expanded[i].position(cx, cy))
		if centroidSign == 0 {
			continue
		}

		maxDist := 0.0
		var farthest [2]float64
		found := false
		for _, p := range hull {
			pos := expanded[i].position(p[0], p[1])
			if signF64(pos) != centroidSign {
				d := expanded[i].distance(p[0], p[1])
				if d > maxDist {
					maxDist = d
					farthest = p
					found = true
				}
			}
		}

		if found {
			expanded[i].moveToPoint(farthest[0], farthest[1])
		}
	}

	return expanded
}

// linesToQuad computes the 4 intersection points of consecutive line pairs.
func linesToQuad(lines [4]lineEq) [4][2]float64 {
	var quad [4][2]float64
	for i := 0; i < 4; i++ {
		j := (i + 1) % 4
		if pt, ok := lines[i].intersect(lines[j]); ok {
			quad[i] = pt
		}
	}
	return quad
}

// ----------------------------------------------------------------
// Line equation  Ax + By + C = 0
// ----------------------------------------------------------------

type lineEq struct {
	A, B, C float64
}

func lineFromPoints(x1, y1, x2, y2 float64) lineEq {
	return lineEq{A: y2 - y1, B: x1 - x2, C: x2*y1 - x1*y2}
}

func (l lineEq) norm() float64 { return math.Hypot(l.A, l.B) }

func (l lineEq) position(x, y float64) float64 { return l.A*x + l.B*y + l.C }

func (l lineEq) distance(x, y float64) float64 {
	n := l.norm()
	if n < 1e-12 {
		return 0
	}
	return math.Abs(l.position(x, y)) / n
}

func (l lineEq) intersect(o lineEq) ([2]float64, bool) {
	det := l.A*o.B - o.A*l.B
	if math.Abs(det) < 1e-12 {
		return [2]float64{}, false
	}
	return [2]float64{
		(-l.C*o.B + o.C*l.B) / det,
		(-l.A*o.C + o.A*l.C) / det,
	}, true
}

func (l *lineEq) moveToPoint(x, y float64) {
	l.C = -(l.A*x + l.B*y)
}

// ----------------------------------------------------------------
// Polygon geometry (float64)
// ----------------------------------------------------------------

// polyArea computes area using the shoelace formula.
func polyArea(pts [][2]float64) float64 {
	n := len(pts)
	if n < 3 {
		return 0
	}
	s := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		s += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]
	}
	return math.Abs(s) / 2
}

// polyIoU computes Intersection over Union of two convex polygons.
func polyIoU(p1, p2 [][2]float64, p1Area float64) float64 {
	inter := sutherlandHodgman(p1, p2)
	if len(inter) < 3 {
		return 0
	}
	intArea := polyArea(inter)
	p2Area := polyArea(p2)
	union := p1Area + p2Area - intArea
	if union <= 0 {
		return 0
	}
	return intArea / union
}

// sutherlandHodgman clips subject polygon by clip polygon.
// Both must be convex with consistent winding (CCW).
func sutherlandHodgman(subject, clip [][2]float64) [][2]float64 {
	output := make([][2]float64, len(subject))
	copy(output, subject)

	for i := range clip {
		if len(output) == 0 {
			return nil
		}
		input := output
		output = make([][2]float64, 0, len(input)+2)

		a := clip[i]
		b := clip[(i+1)%len(clip)]

		for j := range input {
			cur := input[j]
			prev := input[(j+len(input)-1)%len(input)]

			curIn := leftOf(a, b, cur) >= 0
			prevIn := leftOf(a, b, prev) >= 0

			if curIn {
				if !prevIn {
					output = append(output, segIsect(prev, cur, a, b))
				}
				output = append(output, cur)
			} else if prevIn {
				output = append(output, segIsect(prev, cur, a, b))
			}
		}
	}

	return output
}

func leftOf(a, b, c [2]float64) float64 {
	return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
}

func segIsect(p1, p2, p3, p4 [2]float64) [2]float64 {
	d1x, d1y := p2[0]-p1[0], p2[1]-p1[1]
	d2x, d2y := p4[0]-p3[0], p4[1]-p3[1]
	denom := d1x*d2y - d1y*d2x
	if math.Abs(denom) < 1e-12 {
		return [2]float64{(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2}
	}
	t := ((p3[0]-p1[0])*d2y - (p3[1]-p1[1])*d2x) / denom
	return [2]float64{p1[0] + t*d1x, p1[1] + t*d1y}
}

// ----------------------------------------------------------------
// Convex hull (Graham scan, CCW winding)
// ----------------------------------------------------------------

func convexHull(points [][2]float32) [][2]float32 {
	if len(points) < 3 {
		return points
	}

	// Find bottom-most (then left-most) point
	minIdx := 0
	for i := 1; i < len(points); i++ {
		if points[i][1] < points[minIdx][1] ||
			(points[i][1] == points[minIdx][1] && points[i][0] < points[minIdx][0]) {
			minIdx = i
		}
	}

	pts := make([][2]float32, len(points))
	copy(pts, points)
	pts[0], pts[minIdx] = pts[minIdx], pts[0]
	pivot := pts[0]

	sorted := make([][2]float32, len(pts)-1)
	copy(sorted, pts[1:])
	sort.Slice(sorted, func(i, j int) bool {
		a1 := math.Atan2(float64(sorted[i][1]-pivot[1]), float64(sorted[i][0]-pivot[0]))
		a2 := math.Atan2(float64(sorted[j][1]-pivot[1]), float64(sorted[j][0]-pivot[0]))
		if a1 != a2 {
			return a1 < a2
		}
		return squaredDist32(pivot, sorted[i]) < squaredDist32(pivot, sorted[j])
	})

	hull := make([][2]float32, 0, len(pts))
	hull = append(hull, pivot)
	for _, p := range sorted {
		for len(hull) > 1 && cross32(hull[len(hull)-2], hull[len(hull)-1], p) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}

	return hull
}

// ----------------------------------------------------------------
// Douglas-Peucker simplification for closed polygon (ring)
// ----------------------------------------------------------------

// simplifyHullForFit progressively simplifies the hull until it has ≤ maxSides vertices.
func simplifyHullForFit(hull [][2]float64, hullArea float64) [][2]float64 {
	if len(hull) <= fitterMaxHullSides {
		return hull
	}

	simplified := hull
	epsilon := fitterStartEpsilon
	for epsilon <= fitterMaxEpsilon {
		candidate := simplifyRingDP(simplified, epsilon)
		if len(candidate) < 4 {
			break
		}
		if len(candidate) <= fitterMaxHullSides {
			iou := polyIoU(hull, candidate, hullArea)
			if iou > fitterIoUThreshold {
				return candidate
			}
			return simplified
		}
		simplified = candidate
		epsilon += fitterEpsilonStep
	}

	return simplified
}

// simplifyRingDP applies Douglas-Peucker to a closed polygon.
func simplifyRingDP(ring [][2]float64, epsilon float64) [][2]float64 {
	n := len(ring)
	if n <= 4 {
		return ring
	}

	// Find two farthest points as split anchors
	maxDist := 0.0
	farIdx := 0
	for i := 1; i < n; i++ {
		d := distF64(ring[0], ring[i])
		if d > maxDist {
			maxDist = d
			farIdx = i
		}
	}

	// Split into two paths and simplify each
	path1 := make([][2]float64, farIdx+1)
	copy(path1, ring[:farIdx+1])

	path2Len := n - farIdx + 1
	path2 := make([][2]float64, path2Len)
	copy(path2, ring[farIdx:])
	path2[path2Len-1] = ring[0]

	sp1 := simplifyOpenDP(path1, epsilon)
	sp2 := simplifyOpenDP(path2, epsilon)

	result := make([][2]float64, 0, len(sp1)+len(sp2)-2)
	result = append(result, sp1[:len(sp1)-1]...)
	result = append(result, sp2[:len(sp2)-1]...)

	return result
}

// simplifyOpenDP applies Douglas-Peucker to an open polyline.
func simplifyOpenDP(pts [][2]float64, epsilon float64) [][2]float64 {
	if len(pts) <= 2 {
		return pts
	}

	maxDist := 0.0
	maxIdx := 0
	for i := 1; i < len(pts)-1; i++ {
		d := ptSegDist(pts[i], pts[0], pts[len(pts)-1])
		if d > maxDist {
			maxDist = d
			maxIdx = i
		}
	}

	if maxDist > epsilon {
		left := simplifyOpenDP(pts[:maxIdx+1], epsilon)
		right := simplifyOpenDP(pts[maxIdx:], epsilon)
		result := make([][2]float64, 0, len(left)+len(right)-1)
		result = append(result, left[:len(left)-1]...)
		result = append(result, right...)
		return result
	}

	return [][2]float64{pts[0], pts[len(pts)-1]}
}

// ----------------------------------------------------------------
// Nelder-Mead optimizer for 3 parameters
// ----------------------------------------------------------------

func nelderMead3(fn func([3]float64) float64, x0 [3]float64, maxIter int) [3]float64 {
	const (
		alpha = 1.0 // reflection
		gamma = 2.0 // expansion
		rho   = 0.5 // contraction
		sigma = 0.5 // shrink
	)

	var s [4][3]float64
	s[0] = x0
	for i := 0; i < 3; i++ {
		s[i+1] = x0
		if math.Abs(x0[i]) > 1e-6 {
			s[i+1][i] *= 1.05
		} else {
			s[i+1][i] = 0.00025
		}
	}

	var fv [4]float64
	for i := range s {
		fv[i] = fn(s[i])
	}

	for iter := 0; iter < maxIter; iter++ {
		// Sort by function value
		idx := [4]int{0, 1, 2, 3}
		sort.Slice(idx[:], func(a, b int) bool { return fv[idx[a]] < fv[idx[b]] })
		var ss [4][3]float64
		var sf [4]float64
		for i, ii := range idx {
			ss[i] = s[ii]
			sf[i] = fv[ii]
		}
		s, fv = ss, sf

		// Centroid of the 3 best vertices
		var c [3]float64
		for i := 0; i < 3; i++ {
			for d := 0; d < 3; d++ {
				c[d] += s[i][d]
			}
		}
		for d := range c {
			c[d] /= 3
		}

		// Reflection
		var xr [3]float64
		for d := range xr {
			xr[d] = c[d] + alpha*(c[d]-s[3][d])
		}
		fr := fn(xr)

		if fr >= fv[0] && fr < fv[2] {
			s[3], fv[3] = xr, fr
			continue
		}

		// Expansion
		if fr < fv[0] {
			var xe [3]float64
			for d := range xe {
				xe[d] = c[d] + gamma*(xr[d]-c[d])
			}
			if fe := fn(xe); fe < fr {
				s[3], fv[3] = xe, fe
			} else {
				s[3], fv[3] = xr, fr
			}
			continue
		}

		// Contraction
		var xc [3]float64
		for d := range xc {
			xc[d] = c[d] + rho*(s[3][d]-c[d])
		}
		if fc := fn(xc); fc < fv[3] {
			s[3], fv[3] = xc, fc
			continue
		}

		// Shrink toward best
		for i := 1; i < 4; i++ {
			for d := 0; d < 3; d++ {
				s[i][d] = s[0][d] + sigma*(s[i][d]-s[0][d])
			}
			fv[i] = fn(s[i])
		}
	}

	best := 0
	for i := 1; i < 4; i++ {
		if fv[i] < fv[best] {
			best = i
		}
	}
	return s[best]
}

// ----------------------------------------------------------------
// Ordering and conversion helpers
// ----------------------------------------------------------------

// orderClockwiseF64 orders quad vertices clockwise starting from top-left (image coords, Y down).
func orderClockwiseF64(quad [4][2]float64) [4][2]float64 {
	var cx, cy float64
	for _, p := range quad {
		cx += p[0]
		cy += p[1]
	}
	cx /= 4
	cy /= 4

	pts := quad[:]
	sort.Slice(pts, func(i, j int) bool {
		return math.Atan2(pts[i][1]-cy, pts[i][0]-cx) < math.Atan2(pts[j][1]-cy, pts[j][0]-cx)
	})

	// Rotate so top-left (min x+y) is first
	minSum := math.MaxFloat64
	startIdx := 0
	for i, p := range pts {
		if s := p[0] + p[1]; s < minSum {
			minSum = s
			startIdx = i
		}
	}

	var result [4][2]float64
	for i := 0; i < 4; i++ {
		result[i] = pts[(startIdx+i)%4]
	}
	return result
}

// hullToQuadF64 converts a hull with ≤4 points to a quadrilateral.
func hullToQuadF64(hull [][2]float64) [4][2]float64 {
	var q [4][2]float64
	switch len(hull) {
	case 0:
		return q
	case 1:
		p := hull[0]
		q = [4][2]float64{
			{p[0] - 1, p[1] - 1}, {p[0] + 1, p[1] - 1},
			{p[0] + 1, p[1] + 1}, {p[0] - 1, p[1] + 1},
		}
	case 2:
		p1, p2 := hull[0], hull[1]
		l := math.Hypot(p2[0]-p1[0], p2[1]-p1[1])
		if l < 0.001 {
			l = 1
		}
		nx, ny := -(p2[1]-p1[1])/l, (p2[0]-p1[0])/l
		q = [4][2]float64{
			{p1[0] + nx, p1[1] + ny}, {p2[0] + nx, p2[1] + ny},
			{p2[0] - nx, p2[1] - ny}, {p1[0] - nx, p1[1] - ny},
		}
	case 3:
		for i := 0; i < 3; i++ {
			q[i] = hull[i]
		}
		// Fourth point: midpoint of longest edge
		var maxD float64
		for i := 0; i < 3; i++ {
			j := (i + 1) % 3
			if d := distF64(hull[i], hull[j]); d > maxD {
				maxD = d
				q[3] = [2]float64{(hull[i][0] + hull[j][0]) / 2, (hull[i][1] + hull[j][1]) / 2}
			}
		}
	default:
		for i := 0; i < 4 && i < len(hull); i++ {
			q[i] = hull[i]
		}
	}
	return orderClockwiseF64(q)
}

func f32sToF64s(pts [][2]float32) [][2]float64 {
	out := make([][2]float64, len(pts))
	for i, p := range pts {
		out[i] = [2]float64{float64(p[0]), float64(p[1])}
	}
	return out
}

func f64QuadToF32(q [4][2]float64) [4][2]float32 {
	var out [4][2]float32
	for i, p := range q {
		out[i] = [2]float32{float32(p[0]), float32(p[1])}
	}
	return out
}

// ----------------------------------------------------------------
// Low-level geometric helpers
// ----------------------------------------------------------------

func cross32(o, a, b [2]float32) float32 {
	return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
}

func squaredDist32(a, b [2]float32) float32 {
	dx, dy := b[0]-a[0], b[1]-a[1]
	return dx*dx + dy*dy
}

func distF64(a, b [2]float64) float64 {
	return math.Hypot(b[0]-a[0], b[1]-a[1])
}

// ptSegDist computes perpendicular distance from point p to line through a-b.
func ptSegDist(p, a, b [2]float64) float64 {
	dx, dy := b[0]-a[0], b[1]-a[1]
	lenSq := dx*dx + dy*dy
	if lenSq < 1e-12 {
		return distF64(p, a)
	}
	return math.Abs(dx*(a[1]-p[1])-dy*(a[0]-p[0])) / math.Sqrt(lenSq)
}

func signF64(x float64) int {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}
