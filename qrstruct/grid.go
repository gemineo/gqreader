package qrstruct

import (
	"image"
	"math"
)

// Point is an (x, y) position in image pixels.
type Point struct{ X, Y float64 }

// Quad holds four corners in the order top-left, top-right, bottom-right,
// bottom-left of the module lattice (whatever their position in the image).
type Quad [4]Point

// rotate returns the quad with the corner order shifted by k so that a
// different image corner plays the role of the symbol's top-left.
func (q Quad) rotate(k int) Quad {
	var out Quad
	for i := range q {
		out[i] = q[(i+k)%4]
	}
	return out
}

// mirror swaps the top-right and bottom-left corners, which maps the lattice
// through its main diagonal (a mirrored symbol).
func (q Quad) mirror() Quad {
	return Quad{q[0], q[3], q[2], q[1]}
}

func (q Quad) centre() Point {
	var c Point
	for _, p := range q {
		c.X += p.X / 4
		c.Y += p.Y / 4
	}
	return c
}

// Surface is the tilted-axis cylinder model applied in module space before
// the homography. Bend is the half-angle (radians) subtended by the symbol on
// the cylinder, Phase shifts the generatrix facing the camera (-1..1 across
// the symbol) and Theta is the orientation of the bend axis. A zero Bend is
// the identity.
type Surface struct {
	Bend  float64
	Phase float64
	Theta float64
}

// homography maps the module lattice square [0,n]^2 to the quad.
// It returns the 3x3 matrix in row-major order.
func homography(n float64, q Quad) [9]float64 {
	src := [4]Point{{0, 0}, {n, 0}, {n, n}, {0, n}}
	// Solve the 8x8 system for h11..h32 (h33 = 1).
	var a [8][9]float64
	for i := 0; i < 4; i++ {
		sx, sy := src[i].X, src[i].Y
		dx, dy := q[i].X, q[i].Y
		a[2*i] = [9]float64{sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy, dx}
		a[2*i+1] = [9]float64{0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy, dy}
	}
	// Gaussian elimination with partial pivoting.
	for col := 0; col < 8; col++ {
		piv := col
		for r := col + 1; r < 8; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[piv][col]) {
				piv = r
			}
		}
		a[col], a[piv] = a[piv], a[col]
		p := a[col][col]
		if math.Abs(p) < 1e-12 {
			continue
		}
		for r := 0; r < 8; r++ {
			if r == col {
				continue
			}
			f := a[r][col] / p
			if f == 0 {
				continue
			}
			for c := col; c < 9; c++ {
				a[r][c] -= f * a[col][c]
			}
		}
	}
	var h [9]float64
	for i := 0; i < 8; i++ {
		if a[i][i] != 0 {
			h[i] = a[i][8] / a[i][i]
		}
	}
	h[8] = 1
	return h
}

// sampler evaluates a grid model on a grayscale image for one symbol size.
type sampler struct {
	img *image.Gray
	n   int
	// offsets are the sub-sample offsets (in modules) around each centre.
	offsets []float64
	// buffers reused across evaluations.
	vals []float32
}

func newSampler(img *image.Gray, n int, subSamples int, spread float64) *sampler {
	s := &sampler{img: img, n: n, vals: make([]float32, n*n)}
	if subSamples <= 1 {
		s.offsets = []float64{0}
	} else {
		for i := 0; i < subSamples; i++ {
			s.offsets = append(s.offsets, -spread+2*spread*float64(i)/float64(subSamples-1))
		}
	}
	return s
}

// bilinear samples the image at (x, y) with edge replication.
func (s *sampler) bilinear(x, y float64) float32 {
	w, h := s.img.Rect.Dx(), s.img.Rect.Dy()
	if x < 0 {
		x = 0
	} else if x > float64(w-1) {
		x = float64(w - 1)
	}
	if y < 0 {
		y = 0
	} else if y > float64(h-1) {
		y = float64(h - 1)
	}
	x0 := int(x)
	y0 := int(y)
	x1 := x0 + 1
	y1 := y0 + 1
	if x1 >= w {
		x1 = w - 1
	}
	if y1 >= h {
		y1 = h - 1
	}
	fx := float32(x - float64(x0))
	fy := float32(y - float64(y0))
	pix := s.img.Pix
	stride := s.img.Stride
	p00 := float32(pix[y0*stride+x0])
	p01 := float32(pix[y0*stride+x1])
	p10 := float32(pix[y1*stride+x0])
	p11 := float32(pix[y1*stride+x1])
	top := p00 + (p01-p00)*fx
	bot := p10 + (p11-p10)*fx
	return top + (bot-top)*fy
}

// values samples every module centre through the model and returns the n*n
// averaged intensities (row-major). The returned slice is owned by the
// sampler and overwritten by the next call.
func (s *sampler) values(q Quad, surf Surface) []float32 {
	n := s.n
	nf := float64(n)
	h := homography(nf, q)
	for i := range s.vals {
		s.vals[i] = 0
	}
	ct, st := math.Cos(surf.Theta), math.Sin(surf.Theta)
	bend := surf.Bend
	useSurf := math.Abs(bend) > 1e-6
	var fLo, fHi float64
	if useSurf {
		fLo = math.Sin(bend * (-1 - surf.Phase))
		fHi = math.Sin(bend * (1 - surf.Phase))
	}
	half := nf / 2
	inv := 1 / float32(len(s.offsets)*len(s.offsets))
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			var acc float32
			for _, dv := range s.offsets {
				for _, du := range s.offsets {
					u := float64(c) + 0.5 + du
					v := float64(r) + 0.5 + dv
					if useSurf {
						x := u - half
						y := v - half
						xr := ct*x + st*y
						yr := -st*x + ct*y
						f := math.Sin(bend * (xr/half - surf.Phase))
						xb := half * (2*(f-fLo)/(fHi-fLo) - 1)
						u = half + ct*xb - st*yr
						v = half + st*xb + ct*yr
					}
					den := h[6]*u + h[7]*v + h[8]
					px := (h[0]*u + h[1]*v + h[2]) / den
					py := (h[3]*u + h[4]*v + h[5]) / den
					acc += s.bilinear(px, py)
				}
			}
			s.vals[r*n+c] = acc * inv
		}
	}
	return s.vals
}

// otsu returns the threshold maximising the between-class variance of the
// sampled values (clipped to 0..255).
func otsu(vals []float32) float32 {
	var hist [256]int
	for _, v := range vals {
		i := int(v + 0.5)
		if i < 0 {
			i = 0
		} else if i > 255 {
			i = 255
		}
		hist[i]++
	}
	total := len(vals)
	var sum float64
	for i, c := range hist {
		sum += float64(i * c)
	}
	var sumB float64
	wB := 0
	best := -1.0
	thr := 0
	for i, c := range hist {
		wB += c
		if wB == 0 {
			continue
		}
		wF := total - wB
		if wF == 0 {
			break
		}
		sumB += float64(i * c)
		mB := sumB / float64(wB)
		mF := (sum - sumB) / float64(wF)
		between := float64(wB) * float64(wF) * (mB - mF) * (mB - mF)
		if between > best {
			best = between
			thr = i
		}
	}
	// Otsu's threshold is inclusive of the bin; dark = v < thr+1 matches the
	// usual cv::threshold convention (v > thr is light).
	return float32(thr) + 1
}

// binarize fills bits with 1 for dark modules and returns the threshold.
func binarize(vals []float32, bits []uint8) float32 {
	t := otsu(vals)
	for i, v := range vals {
		if v < t {
			bits[i] = 1
		} else {
			bits[i] = 0
		}
	}
	return t
}
