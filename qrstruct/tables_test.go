package qrstruct

import (
	"reflect"
	"testing"
)

// totalCodewordsRef is ISO 18004 Table 1 (total number of codewords per version).
var totalCodewordsRef = [MaxVersion]int{
	26, 44, 70, 100, 134, 172, 196, 242, 292, 346, 404, 466, 532, 581, 655, 733, 815, 901, 991, 1085,
	1156, 1258, 1364, 1474, 1588, 1706, 1828, 1921, 2051, 2185, 2323, 2465, 2611, 2761, 2876, 3034, 3196, 3362, 3532, 3706,
}

func TestTotalCodewords(t *testing.T) {
	for v := MinVersion; v <= MaxVersion; v++ {
		if got, want := TotalCodewords(v), totalCodewordsRef[v-1]; got != want {
			t.Errorf("version %d: TotalCodewords = %d, want %d", v, got, want)
		}
	}
}

func TestBlockLayoutSums(t *testing.T) {
	for v := MinVersion; v <= MaxVersion; v++ {
		for lvl := LevelL; lvl <= LevelH; lvl++ {
			blocks := BlockLayout(v, lvl)
			sum := 0
			for _, b := range blocks {
				sum += b.Data + b.EC
				if b.Data <= 0 || b.EC <= 0 {
					t.Errorf("version %d level %s: degenerate block %+v", v, lvl, b)
				}
			}
			if sum != TotalCodewords(v) {
				t.Errorf("version %d level %s: blocks sum to %d, want %d", v, lvl, sum, TotalCodewords(v))
			}
		}
	}
}

func TestDataPositionsCoverCodewords(t *testing.T) {
	for v := MinVersion; v <= MaxVersion; v++ {
		l := LayoutFor(v)
		got := len(l.dataPos)
		want := rawDataModules(v)
		if got != want {
			t.Errorf("version %d: %d data modules, want %d", v, got, want)
		}
		if got/8 != TotalCodewords(v) {
			t.Errorf("version %d: %d whole codewords, want %d", v, got/8, TotalCodewords(v))
		}
		// Every data position must be unique and outside the function area.
		seen := make(map[[2]int]bool, got)
		for _, p := range l.dataPos {
			if seen[p] {
				t.Errorf("version %d: duplicate data position %v", v, p)
			}
			seen[p] = true
			if l.isFunction[p[0]*l.Size+p[1]] {
				t.Errorf("version %d: data position %v is a function module", v, p)
			}
		}
	}
}

func TestAlignmentPositions(t *testing.T) {
	tests := []struct {
		version int
		want    []int
	}{
		{1, nil},
		{2, []int{6, 18}},
		{6, []int{6, 34}},
		{7, []int{6, 22, 38}},
		{14, []int{6, 26, 46, 66}},
		{21, []int{6, 28, 50, 72, 94}},
		{32, []int{6, 34, 60, 86, 112, 138}},
		{40, []int{6, 30, 58, 86, 114, 142, 170}},
	}
	for _, tc := range tests {
		if got := alignmentPositions(tc.version); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("version %d: alignment positions = %v, want %v", tc.version, got, tc.want)
		}
	}
}

func TestFormatBits(t *testing.T) {
	// ISO 18004 Annex C example: level M, mask 5 -> 100000011001110.
	if got := formatBits(LevelM, 5); got != 0x40CE {
		t.Errorf("formatBits(M,5) = %015b, want %015b", got, 0x40CE)
	}
	// Every valid format word must be at Hamming distance >= 7 from the others.
	for i := range allFormats {
		for j := range allFormats {
			if i != j && popcount(allFormats[i]^allFormats[j]) < 7 {
				t.Errorf("format words %d and %d too close", i, j)
			}
		}
	}
}

func TestVersionBits(t *testing.T) {
	// ISO 18004 Annex D example: version 7 -> 000111110010010100.
	if got := versionBits(7); got != 0x07C94 {
		t.Errorf("versionBits(7) = %018b, want %018b", got, 0x07C94)
	}
}
