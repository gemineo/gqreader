package qrstruct

// This file holds the ISO/IEC 18004 constants needed to lay out a symbol and to
// split its codewords into Reed-Solomon blocks.

// MinVersion and MaxVersion bound the QR code versions supported by the package.
const (
	MinVersion = 1
	MaxVersion = 40
)

// Level is a QR code error correction level.
type Level int

// Error correction levels, ordered by increasing redundancy.
const (
	LevelL Level = iota
	LevelM
	LevelQ
	LevelH
)

// String returns the single-letter name of the level.
func (l Level) String() string {
	switch l {
	case LevelL:
		return "L"
	case LevelM:
		return "M"
	case LevelQ:
		return "Q"
	case LevelH:
		return "H"
	}
	return "?"
}

// formatBits returns the two-bit level indicator used in the format information.
func (l Level) formatBits() int {
	switch l {
	case LevelL:
		return 1
	case LevelM:
		return 0
	case LevelQ:
		return 3
	default:
		return 2
	}
}

// eccCodewordsPerBlock[level][version-1] is the number of error correction
// codewords in every block of a symbol (ISO 18004 Table 9).
var eccCodewordsPerBlock = [4][MaxVersion]int{
	LevelL: {7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30},
	LevelM: {10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28},
	LevelQ: {13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30},
	LevelH: {17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30},
}

// numErrorCorrectionBlocks[level][version-1] is the number of Reed-Solomon
// blocks a symbol is split into (ISO 18004 Table 9).
var numErrorCorrectionBlocks = [4][MaxVersion]int{
	LevelL: {1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25},
	LevelM: {1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49},
	LevelQ: {1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68},
	LevelH: {1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81},
}

// SymbolSize returns the number of modules per side for a version.
func SymbolSize(version int) int {
	return 17 + 4*version
}

// rawDataModules returns the number of modules available for data and error
// correction codewords (everything outside the function patterns).
func rawDataModules(version int) int {
	n := (16*version+128)*version + 64
	if version >= 2 {
		numAlign := version/7 + 2
		n -= (25*numAlign-10)*numAlign - 55
		if version >= 7 {
			n -= 36
		}
	}
	return n
}

// TotalCodewords returns the total number of codewords (data + EC) of a version.
func TotalCodewords(version int) int {
	return rawDataModules(version) / 8
}

// alignmentPositions returns the centre coordinates of the alignment patterns
// along one axis. The same list applies to rows and columns.
func alignmentPositions(version int) []int {
	if version == 1 {
		return nil
	}
	numAlign := version/7 + 2
	var step int
	if version == 32 {
		step = 26
	} else {
		step = (version*4 + numAlign*2 + 1) / (numAlign*2 - 2) * 2
	}
	pos := make([]int, numAlign)
	pos[0] = 6
	size := SymbolSize(version)
	for i := numAlign - 1; i >= 1; i-- {
		pos[i] = size - 7 - (numAlign-1-i)*step
	}
	return pos
}

// Block describes one Reed-Solomon block: the number of data codewords and
// the number of error correction codewords.
type Block struct {
	Data int
	EC   int
}

// BlockLayout returns the Reed-Solomon blocks of a symbol in the order used
// for interleaving: the short blocks first, then the long ones.
func BlockLayout(version int, level Level) []Block {
	total := TotalCodewords(version)
	nb := numErrorCorrectionBlocks[level][version-1]
	ec := eccCodewordsPerBlock[level][version-1]
	shortLen := total / nb
	numLong := total % nb
	blocks := make([]Block, nb)
	for i := range blocks {
		dataLen := shortLen - ec
		if i >= nb-numLong {
			dataLen++
		}
		blocks[i] = Block{Data: dataLen, EC: ec}
	}
	return blocks
}

// DataCodewords returns the number of data codewords of a symbol.
func DataCodewords(version int, level Level) int {
	n := 0
	for _, b := range BlockLayout(version, level) {
		n += b.Data
	}
	return n
}

// deinterleaveIndex returns, for every block, the positions of its codewords
// in the interleaved codeword stream (data codewords first, then EC codewords,
// each interleaved block by block as described in ISO 18004 §7.6).
func deinterleaveIndex(blocks []Block) [][]int {
	idx := make([][]int, len(blocks))
	maxData := 0
	for i, b := range blocks {
		idx[i] = make([]int, b.Data+b.EC)
		if b.Data > maxData {
			maxData = b.Data
		}
	}
	p := 0
	for k := 0; k < maxData; k++ {
		for i, b := range blocks {
			if k < b.Data {
				idx[i][k] = p
				p++
			}
		}
	}
	ec := blocks[0].EC
	for k := 0; k < ec; k++ {
		for i, b := range blocks {
			idx[i][b.Data+k] = p
			p++
		}
	}
	return idx
}
