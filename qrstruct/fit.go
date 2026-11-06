package qrstruct

import (
	"image"
	"math"
	"sort"
)

// evaluator scores grid hypotheses for one (image, version) pair.
type evaluator struct {
	layout  *Layout
	sampler *sampler
	bits    []uint8
}

func newEvaluator(img *image.Gray, version int, subSamples int) *evaluator {
	l := LayoutFor(version)
	return &evaluator{
		layout:  l,
		sampler: newSampler(img, l.Size, subSamples, 0.2),
		bits:    make([]uint8, l.Size*l.Size),
	}
}

// knownScore counts the a-priori-known modules that disagree with the
// binarised samples, plus the Hamming distance of the format information to
// the closest valid format word.
func (e *evaluator) knownScore(bits []uint8) int {
	mism := 0
	for i, k := range e.layout.known {
		if k >= 0 && int8(bits[i]) != k {
			mism++
		}
	}
	_, _, d := e.layout.readFormat(bits)
	return mism + d
}

// ambiguity is the fraction of sampled values that fall inside a band around
// the threshold. A well-aligned grid yields bimodal samples on every module,
// known or not, so this term extends the objective to the data area. The
// band is 15 % of the 10th-90th percentile range, read off a 256-bin
// histogram.
func ambiguity(vals []float32, t float32) float64 {
	var hist [256]int
	for _, v := range vals {
		hist[clampByte(v)]++
	}
	n := len(vals)
	p10, p90 := n/10, n*9/10
	lo, hi := -1, -1
	acc := 0
	for i, c := range hist {
		acc += c
		if lo < 0 && acc > p10 {
			lo = i
		}
		if acc > p90 {
			hi = i
			break
		}
	}
	if hi < 0 {
		hi = 255
	}
	band := 0.15 * float32(hi-lo)
	cnt := 0
	for _, v := range vals {
		d := v - t
		if d < 0 {
			d = -d
		}
		if d < band {
			cnt++
		}
	}
	return float64(cnt) / float64(n)
}

func clampByte(v float32) int {
	i := int(v + 0.5)
	if i < 0 {
		return 0
	}
	if i > 255 {
		return 255
	}
	return i
}

// objective evaluates a hypothesis. lambda weights the ambiguity term
// (0 disables it).
func (e *evaluator) objective(q Quad, surf Surface, lambda float64) float64 {
	vals := e.sampler.values(q, surf)
	t := binarize(vals, e.bits)
	s := float64(e.knownScore(e.bits))
	if lambda > 0 {
		s += lambda * ambiguity(vals, t)
	}
	return s
}

// normalised returns a score comparable across versions.
func (e *evaluator) normalised(score float64) float64 {
	return score / float64(e.layout.numKnown)
}

// sideLength returns the mean edge length of the quad in pixels.
func sideLength(q Quad) float64 {
	s := 0.0
	for i := 0; i < 4; i++ {
		a, b := q[i], q[(i+1)%4]
		s += math.Hypot(b.X-a.X, b.Y-a.Y)
	}
	return s / 4
}

// globalSearch slides and scales the quad as a whole. Shifts span ±5 % of
// the side length, and at least ±6 px so that small symbols tolerate the same
// absolute detector error as large ones, on a 9×9 grid. Candidates are
// visited from the identity outwards so that, on equal scores, the placement
// closest to the input quad wins.
//
// It returns the best placement overall and the best placement within the
// inner half of the window. The two can sit in different basins of the
// objective, and a descent from the far one is not always the better one, so
// callers descend from both.
func (e *evaluator) globalSearch(q Quad, lambda float64) (wide, near Quad) {
	side := sideLength(q)
	step := math.Max(6.0, side*0.05) / 4
	c := q.centre()
	bestWide, bestNear := math.Inf(1), math.Inf(1)
	wide, near = q, q
	for _, scale := range []float64{1.0, 0.94, 1.06} {
		var base Quad
		for i, p := range q {
			base[i] = Point{c.X + (p.X-c.X)*scale, c.Y + (p.Y-c.Y)*scale}
		}
		for _, sh := range globalShifts {
			var cand Quad
			for i, p := range base {
				cand[i] = Point{p.X + float64(sh[0])*step, p.Y + float64(sh[1])*step}
			}
			s := e.objective(cand, Surface{}, lambda)
			if s < bestWide {
				bestWide, wide = s, cand
			}
			if abs(sh[0]) <= 2 && abs(sh[1]) <= 2 && s < bestNear {
				bestNear, near = s, cand
			}
		}
	}
	return wide, near
}

// globalShifts lists the (dx, dy) grid of the global search ordered by
// distance from the origin.
var globalShifts = func() [][2]int {
	var out [][2]int
	for dx := -4; dx <= 4; dx++ {
		for dy := -4; dy <= 4; dy++ {
			out = append(out, [2]int{dx, dy})
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		return out[i][0]*out[i][0]+out[i][1]*out[i][1] < out[j][0]*out[j][0]+out[j][1]*out[j][1]
	})
	return out
}()

// descent runs a coordinate descent on the 8 corner coordinates with the
// given step sizes (fractions of the side length, clamped to 0.1 px). At each
// step size, whole-quad moves (translation and scaling) are tried as well,
// which helps escape local minima where every corner is off in the same way.
func (e *evaluator) descent(q Quad, surf Surface, stepFracs []float64, lambda float64) (Quad, float64) {
	side := sideLength(q)
	best := e.objective(q, surf, lambda)
	for _, frac := range stepFracs {
		step := math.Max(0.1, side*frac)
		for improved := true; improved; {
			improved = false
			for i := 0; i < 4; i++ {
				for axis := 0; axis < 2; axis++ {
					for _, sgn := range []float64{-1, 1} {
						cand := q
						if axis == 0 {
							cand[i].X += sgn * step
						} else {
							cand[i].Y += sgn * step
						}
						if s := e.objective(cand, surf, lambda); s < best {
							best, q, improved = s, cand, true
						}
					}
				}
			}
			for _, mv := range quadMoves(q, step) {
				if s := e.objective(mv, surf, lambda); s < best {
					best, q, improved = s, mv, true
				}
			}
		}
	}
	return q, best
}

// quadMoves returns whole-quad variants: four translations and two scalings
// about the centre, each by step pixels.
func quadMoves(q Quad, step float64) []Quad {
	c := q.centre()
	side := sideLength(q)
	out := make([]Quad, 0, 6)
	for _, d := range []Point{{step, 0}, {-step, 0}, {0, step}, {0, -step}} {
		var m Quad
		for i, p := range q {
			m[i] = Point{p.X + d.X, p.Y + d.Y}
		}
		out = append(out, m)
	}
	for _, sgn := range []float64{1, -1} {
		scale := 1 + sgn*step/side
		var m Quad
		for i, p := range q {
			m[i] = Point{c.X + (p.X-c.X)*scale, c.Y + (p.Y-c.Y)*scale}
		}
		out = append(out, m)
	}
	return out
}

// planarLambda weights the ambiguity term during the planar stages. It is
// small enough not to disturb the ranking of hypotheses, and it breaks the
// ties of the integer mismatch count in favour of grids whose samples sit far
// from the threshold, i.e. on module centres.
const planarLambda = 10.0

// Step schedules as fractions of the side length. For a 113 px symbol the
// coarse schedule is about (2.8, 1.4) px and the fine one (1.4 .. 0.17) px.
var (
	coarseSteps = []float64{0.025, 0.0125}
	fineSteps   = []float64{0.0125, 0.006, 0.003, 0.0018, 0.001}
)
