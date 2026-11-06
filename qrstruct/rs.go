package qrstruct

import (
	"errors"
	"fmt"
)

// Reed-Solomon decoder over GF(2^8) with the QR code parameters: primitive
// polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11d), generator element 2, first
// consecutive root alpha^0. Polynomials are stored most significant
// coefficient first. Erasures (known error positions) are supported: a block
// with e errors and s erasures is correctable when 2e + s <= nsym.

var (
	gfExp [512]byte
	gfLog [256]byte
)

func init() {
	x := 1
	for i := 0; i < 255; i++ {
		gfExp[i] = byte(x)
		gfLog[x] = byte(i)
		x <<= 1
		if x&0x100 != 0 {
			x ^= 0x11d
		}
	}
	for i := 255; i < 512; i++ {
		gfExp[i] = gfExp[i-255]
	}
}

func gfMul(a, b byte) byte {
	if a == 0 || b == 0 {
		return 0
	}
	return gfExp[int(gfLog[a])+int(gfLog[b])]
}

func gfDiv(a, b byte) byte {
	if a == 0 {
		return 0
	}
	return gfExp[(int(gfLog[a])+255-int(gfLog[b]))%255]
}

func gfInv(x byte) byte {
	return gfExp[255-int(gfLog[x])]
}

// gfPow2 returns 2^p for any integer p (negative exponents allowed).
func gfPow2(p int) byte {
	return gfExp[((p%255)+255)%255]
}

func polyScale(p []byte, x byte) []byte {
	out := make([]byte, len(p))
	for i, c := range p {
		out[i] = gfMul(c, x)
	}
	return out
}

func polyAdd(p, q []byte) []byte {
	n := max(len(p), len(q))
	out := make([]byte, n)
	for i, c := range p {
		out[i+n-len(p)] = c
	}
	for i, c := range q {
		out[i+n-len(q)] ^= c
	}
	return out
}

func polyMul(p, q []byte) []byte {
	out := make([]byte, len(p)+len(q)-1)
	for j, qc := range q {
		if qc == 0 {
			continue
		}
		for i, pc := range p {
			out[i+j] ^= gfMul(pc, qc)
		}
	}
	return out
}

func polyEval(p []byte, x byte) byte {
	y := p[0]
	for i := 1; i < len(p); i++ {
		y = gfMul(y, x) ^ p[i]
	}
	return y
}

// rsSyndromes returns the nsym syndromes of msg, prefixed by a zero so that
// index i holds S_i for i >= 1.
func rsSyndromes(msg []byte, nsym int) []byte {
	synd := make([]byte, nsym+1)
	for i := 0; i < nsym; i++ {
		synd[i+1] = polyEval(msg, gfPow2(i))
	}
	return synd
}

func allZero(p []byte) bool {
	for _, c := range p {
		if c != 0 {
			return false
		}
	}
	return true
}

// rsForneySyndromes removes the contribution of the erasures from the
// syndromes so that Berlekamp-Massey only has to locate the unknown errors.
func rsForneySyndromes(synd []byte, erasePos []int, nmess int) []byte {
	fsynd := make([]byte, len(synd)-1)
	copy(fsynd, synd[1:])
	for _, p := range erasePos {
		x := gfPow2(nmess - 1 - p)
		for j := 0; j < len(fsynd)-1; j++ {
			fsynd[j] = gfMul(fsynd[j], x) ^ fsynd[j+1]
		}
	}
	return fsynd
}

// rsErrorLocator runs Berlekamp-Massey on the Forney syndromes and returns
// the error locator polynomial.
func rsErrorLocator(fsynd []byte, nsym, eraseCount int) ([]byte, error) {
	errLoc := []byte{1}
	oldLoc := []byte{1}
	for i := 0; i < nsym-eraseCount; i++ {
		k := i
		delta := fsynd[k]
		for j := 1; j < len(errLoc); j++ {
			if k-j < 0 {
				break
			}
			delta ^= gfMul(errLoc[len(errLoc)-1-j], fsynd[k-j])
		}
		oldLoc = append(oldLoc, 0)
		if delta != 0 {
			if len(oldLoc) > len(errLoc) {
				newLoc := polyScale(oldLoc, delta)
				oldLoc = polyScale(errLoc, gfInv(delta))
				errLoc = newLoc
			}
			errLoc = polyAdd(errLoc, polyScale(oldLoc, delta))
		}
	}
	for len(errLoc) > 0 && errLoc[0] == 0 {
		errLoc = errLoc[1:]
	}
	errs := len(errLoc) - 1
	if (errs-eraseCount)*2+eraseCount > nsym {
		return nil, errors.New("too many errors to correct")
	}
	return errLoc, nil
}

// rsFindErrors locates the roots of the error locator (Chien search) and
// returns the corresponding message positions.
func rsFindErrors(errLoc []byte, nmess int) ([]int, error) {
	errs := len(errLoc) - 1
	rev := make([]byte, len(errLoc))
	for i, c := range errLoc {
		rev[len(errLoc)-1-i] = c
	}
	var pos []int
	for i := 0; i < nmess; i++ {
		if polyEval(rev, gfPow2(i)) == 0 {
			pos = append(pos, nmess-1-i)
		}
	}
	if len(pos) != errs {
		return nil, errors.New("error locator degree does not match its root count")
	}
	return pos, nil
}

// rsCorrectErrata computes the error magnitudes (Forney algorithm) at the
// given positions and applies them to msg in place.
func rsCorrectErrata(msg, synd []byte, errPos []int) error {
	nmess := len(msg)
	coefPos := make([]int, len(errPos))
	for i, p := range errPos {
		coefPos[i] = nmess - 1 - p
	}
	// Errata locator: prod (1 + x * 2^coefPos).
	errLoc := []byte{1}
	for _, cp := range coefPos {
		errLoc = polyMul(errLoc, polyAdd([]byte{1}, []byte{gfPow2(cp), 0}))
	}
	// Error evaluator: (reversed syndromes * locator) mod x^(len(errLoc)).
	rsynd := make([]byte, len(synd))
	for i, c := range synd {
		rsynd[len(synd)-1-i] = c
	}
	prod := polyMul(rsynd, errLoc)
	nsym := len(errLoc) - 1
	rem := prod[len(prod)-(nsym+1):]
	errEval := make([]byte, len(rem))
	for i, c := range rem {
		errEval[len(rem)-1-i] = c
	}
	// Root inverses X_i = 2^-(255 - coefPos) = 2^coefPos.
	xs := make([]byte, len(coefPos))
	for i, cp := range coefPos {
		xs[i] = gfPow2(-(255 - cp))
	}
	for i, xi := range xs {
		xiInv := gfInv(xi)
		// Formal derivative of the locator at X_i^-1.
		errLocPrime := byte(1)
		for j, xj := range xs {
			if j != i {
				errLocPrime = gfMul(errLocPrime, 1^gfMul(xiInv, xj))
			}
		}
		if errLocPrime == 0 {
			return errors.New("could not find error magnitude")
		}
		rev := make([]byte, len(errEval))
		for k, c := range errEval {
			rev[len(errEval)-1-k] = c
		}
		y := polyEval(rev, xiInv)
		y = gfMul(xi, y) // fcr = 0: multiply by X_i^(1-fcr)
		msg[errPos[i]] ^= gfDiv(y, errLocPrime)
	}
	return nil
}

// RSDecode corrects msg (data followed by nsym EC codewords) in place and
// returns the number of errata corrected. erasePos lists positions whose
// value is known to be unreliable. The function reports an error when the
// block is beyond repair.
func RSDecode(msg []byte, nsym int, erasePos []int) (int, error) {
	if len(msg) > 255 {
		return 0, fmt.Errorf("message length %d exceeds 255", len(msg))
	}
	if len(erasePos) > nsym {
		return 0, fmt.Errorf("%d erasures exceed %d EC codewords", len(erasePos), nsym)
	}
	for _, p := range erasePos {
		if p < 0 || p >= len(msg) {
			return 0, fmt.Errorf("erasure position %d out of range", p)
		}
		msg[p] = 0
	}
	synd := rsSyndromes(msg, nsym)
	if allZero(synd) {
		return 0, nil
	}
	fsynd := rsForneySyndromes(synd, erasePos, len(msg))
	errLoc, err := rsErrorLocator(fsynd, nsym, len(erasePos))
	if err != nil {
		return 0, err
	}
	errPos, err := rsFindErrors(errLoc, len(msg))
	if err != nil {
		return 0, err
	}
	all := append(append([]int{}, erasePos...), errPos...)
	if err := rsCorrectErrata(msg, synd, all); err != nil {
		return 0, err
	}
	if !allZero(rsSyndromes(msg, nsym)) {
		return 0, errors.New("message still has non-zero syndromes after correction")
	}
	return len(all), nil
}
