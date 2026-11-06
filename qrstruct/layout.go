package qrstruct

import "sync"

// Module coordinates are (row, col), 0-indexed from the top-left finder pattern.

// Layout caches the per-version symbol structure: which modules are function
// modules, what value the a-priori-known ones hold, and the order in which the
// data modules are read.
type Layout struct {
	Version int
	Size    int

	// isFunction[r*Size+c] is true for every module that is not a data module.
	isFunction []bool

	// known[r*Size+c] is 1 (dark) or 0 (light) for modules whose value does not
	// depend on the level, mask or content, and -1 otherwise. Format information
	// bits are flagged as function modules but left unknown here; see
	// expectedWithFormat.
	known []int8

	// dataPos lists the data module coordinates in codeword bit order
	// (bit 0 = most significant bit of codeword 0).
	dataPos [][2]int

	// fmtPos1 / fmtPos2 hold the two copies of the format information; index
	// i holds bit i (least significant bit first).
	fmtPos1 [15][2]int
	fmtPos2 [15][2]int

	// numKnown counts the modules with a known value, format and version
	// information included (used to normalise fit scores across versions).
	numKnown int
}

var (
	layoutMu    sync.Mutex
	layoutCache = map[int]*Layout{}
)

// LayoutFor returns the cached layout of a version. It panics only on an
// out-of-range version, which callers are expected to validate.
func LayoutFor(version int) *Layout {
	layoutMu.Lock()
	defer layoutMu.Unlock()
	if l, ok := layoutCache[version]; ok {
		return l
	}
	l := buildLayout(version)
	layoutCache[version] = l
	return l
}

func buildLayout(version int) *Layout {
	n := SymbolSize(version)
	l := &Layout{
		Version:    version,
		Size:       n,
		isFunction: make([]bool, n*n),
		known:      make([]int8, n*n),
	}
	for i := range l.known {
		l.known[i] = -1
	}
	set := func(r, c int, v int8) {
		l.isFunction[r*n+c] = true
		l.known[r*n+c] = v
	}

	// Finder patterns with their separators.
	finder := func(r0, c0 int) {
		for r := -1; r <= 7; r++ {
			for c := -1; c <= 7; c++ {
				rr, cc := r0+r, c0+c
				if rr < 0 || rr >= n || cc < 0 || cc >= n {
					continue
				}
				switch {
				case r == -1 || r == 7 || c == -1 || c == 7:
					set(rr, cc, 0) // separator
				case r == 0 || r == 6 || c == 0 || c == 6:
					set(rr, cc, 1) // outer ring
				case r == 1 || r == 5 || c == 1 || c == 5:
					set(rr, cc, 0) // inner ring
				default:
					set(rr, cc, 1) // centre
				}
			}
		}
	}
	finder(0, 0)
	finder(0, n-7)
	finder(n-7, 0)

	// Timing patterns.
	for i := 8; i < n-8; i++ {
		v := int8(0)
		if i%2 == 0 {
			v = 1
		}
		set(6, i, v)
		set(i, 6, v)
	}

	// Alignment patterns. The three positions overlapping the finder patterns
	// are skipped; patterns sitting on the timing lines are kept.
	ap := alignmentPositions(version)
	for _, ar := range ap {
		for _, ac := range ap {
			if (ar <= 8 && ac <= 8) || (ar <= 8 && ac >= n-9) || (ar >= n-9 && ac <= 8) {
				continue
			}
			for r := -2; r <= 2; r++ {
				for c := -2; c <= 2; c++ {
					d := max(abs(r), abs(c))
					v := int8(0)
					if d == 0 || d == 2 {
						v = 1
					}
					set(ar+r, ac+c, v)
				}
			}
		}
	}

	// Format information (two copies) and the always-dark module.
	for i := 0; i < 6; i++ {
		l.fmtPos1[i] = [2]int{i, 8}
	}
	l.fmtPos1[6] = [2]int{7, 8}
	l.fmtPos1[7] = [2]int{8, 8}
	l.fmtPos1[8] = [2]int{8, 7}
	for i := 9; i < 15; i++ {
		l.fmtPos1[i] = [2]int{8, 14 - i}
	}
	for i := 0; i < 8; i++ {
		l.fmtPos2[i] = [2]int{8, n - 1 - i}
	}
	for i := 8; i < 15; i++ {
		l.fmtPos2[i] = [2]int{n - 15 + i, 8}
	}
	for i := 0; i < 15; i++ {
		l.isFunction[l.fmtPos1[i][0]*n+l.fmtPos1[i][1]] = true
		l.isFunction[l.fmtPos2[i][0]*n+l.fmtPos2[i][1]] = true
	}
	set(n-8, 8, 1)

	// Version information (versions 7 and up). Its value is known once the
	// version is hypothesised, so it contributes to the fit score.
	if version >= 7 {
		bits := versionBits(version)
		for i := 0; i < 18; i++ {
			v := int8((bits >> uint(i)) & 1)
			set(i/3, n-11+i%3, v)
			set(n-11+i%3, i/3, v)
		}
	}

	// Data module order (ISO 18004 §7.7.3): two-module wide columns, right to
	// left, zigzagging up then down, skipping the vertical timing column.
	for right := n - 1; right >= 1; right -= 2 {
		if right == 6 {
			right--
		}
		upward := ((right + 1) & 2) == 0
		for vert := 0; vert < n; vert++ {
			y := vert
			if upward {
				y = n - 1 - vert
			}
			for j := 0; j < 2; j++ {
				x := right - j
				if !l.isFunction[y*n+x] {
					l.dataPos = append(l.dataPos, [2]int{y, x})
				}
			}
		}
	}

	for _, k := range l.known {
		if k >= 0 {
			l.numKnown++
		}
	}
	l.numKnown += 30 // the two format information copies

	return l
}

// NumDataCodewordsRaw returns how many whole codewords the data area holds.
func (l *Layout) NumDataCodewordsRaw() int {
	return len(l.dataPos) / 8
}

// versionBits returns the 18-bit version information word (6 data bits plus a
// 12-bit Golay-derived BCH remainder).
func versionBits(version int) int {
	rem := version
	for i := 0; i < 12; i++ {
		rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
	}
	return version<<12 | rem
}

// formatBits returns the 15-bit format information word for a level and mask:
// 5 data bits, a 10-bit BCH remainder, XOR-ed with the fixed mask 0x5412.
func formatBits(level Level, mask int) int {
	data := level.formatBits()<<3 | mask
	rem := data
	for i := 0; i < 10; i++ {
		rem = (rem << 1) ^ ((rem >> 9) * 0x537)
	}
	return (data<<10 | rem) ^ 0x5412
}

// allFormats lists the 32 valid format words indexed by level*8+mask.
var allFormats = func() [32]int {
	var f [32]int
	for lvl := LevelL; lvl <= LevelH; lvl++ {
		for m := 0; m < 8; m++ {
			f[int(lvl)*8+m] = formatBits(lvl, m)
		}
	}
	return f
}()

// readFormat decodes the format information from a bit matrix by picking the
// valid format word closest (in Hamming distance, both copies summed) to what
// was read. It returns the level, the mask and that distance.
func (l *Layout) readFormat(bits []uint8) (Level, int, int) {
	n := l.Size
	v1, v2 := 0, 0
	for i := 0; i < 15; i++ {
		v1 |= int(bits[l.fmtPos1[i][0]*n+l.fmtPos1[i][1]]) << uint(i)
		v2 |= int(bits[l.fmtPos2[i][0]*n+l.fmtPos2[i][1]]) << uint(i)
	}
	bestD, bestI := 1<<30, 0
	for i, f := range allFormats {
		d := popcount(v1^f) + popcount(v2^f)
		if d < bestD {
			bestD, bestI = d, i
		}
	}
	return Level(bestI / 8), bestI % 8, bestD
}

// maskBit reports whether the mask pattern inverts module (r, c).
func maskBit(mask, r, c int) bool {
	switch mask {
	case 0:
		return (r+c)%2 == 0
	case 1:
		return r%2 == 0
	case 2:
		return c%3 == 0
	case 3:
		return (r+c)%3 == 0
	case 4:
		return (r/2+c/3)%2 == 0
	case 5:
		return (r*c)%2+(r*c)%3 == 0
	case 6:
		return ((r*c)%2+(r*c)%3)%2 == 0
	default:
		return ((r+c)%2+(r*c)%3)%2 == 0
	}
}

// extractCodewords unmasks the data modules and packs them into codewords in
// interleaved order. Trailing remainder bits are dropped.
func (l *Layout) extractCodewords(bits []uint8, mask int) []byte {
	n := l.Size
	ncw := len(l.dataPos) / 8
	cw := make([]byte, ncw)
	for i := 0; i < ncw*8; i++ {
		p := l.dataPos[i]
		b := bits[p[0]*n+p[1]]
		if maskBit(mask, p[0], p[1]) {
			b ^= 1
		}
		cw[i>>3] |= b << uint(7-(i&7))
	}
	return cw
}

func popcount(x int) int {
	c := 0
	for x != 0 {
		x &= x - 1
		c++
	}
	return c
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
