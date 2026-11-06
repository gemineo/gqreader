package qrstruct

import (
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

type matrixFixture struct {
	File    string `json:"file"`
	Version int    `json:"version"`
	Level   string `json:"level"`
	Mask    int    `json:"mask"`
	Text    string `json:"text"`
}

func loadGray(t *testing.T, path string) *image.Gray {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("opening %s: %v", path, err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decoding %s: %v", path, err)
	}
	gray := image.NewGray(img.Bounds())
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			gray.Set(x, y, img.At(x, y))
		}
	}
	return gray
}

func levelFromString(s string) Level {
	switch s {
	case "L":
		return LevelL
	case "M":
		return LevelM
	case "Q":
		return LevelQ
	default:
		return LevelH
	}
}

// TestDecodeMatrixFixtures decodes encoder-produced module matrices for all
// 40 versions and 4 levels, which validates the tables, the format and
// version information, the data ordering, the de-interleaving and the
// Reed-Solomon decoder.
func TestDecodeMatrixFixtures(t *testing.T) {
	dir := filepath.Join("testdata", "matrices")
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatalf("reading manifest: %v", err)
	}
	var fixtures []matrixFixture
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatalf("parsing manifest: %v", err)
	}
	if len(fixtures) != MaxVersion*4 {
		t.Fatalf("manifest lists %d fixtures, want %d", len(fixtures), MaxVersion*4)
	}
	for _, fx := range fixtures {
		t.Run(fx.File, func(t *testing.T) {
			img := loadGray(t, filepath.Join(dir, fx.File))
			n := SymbolSize(fx.Version)
			if img.Rect.Dx() != n || img.Rect.Dy() != n {
				t.Fatalf("fixture is %dx%d, want %dx%d", img.Rect.Dx(), img.Rect.Dy(), n, n)
			}
			bits := make([]uint8, n*n)
			for i, p := range img.Pix[:n*n] {
				if p < 128 {
					bits[i] = 1
				}
			}
			// The a-priori-known modules must all agree with the encoder output.
			l := LayoutFor(fx.Version)
			for i, k := range l.known {
				if k >= 0 && int8(bits[i]) != k {
					t.Errorf("known module (%d,%d) = %d, layout expects %d", i/n, i%n, bits[i], k)
				}
			}
			r, err := DecodeMatrix(bits, fx.Version)
			if err != nil {
				t.Fatalf("DecodeMatrix: %v", err)
			}
			if r.Level != levelFromString(fx.Level) || r.Mask != fx.Mask {
				t.Errorf("format = %s/%d, want %s/%d", r.Level, r.Mask, fx.Level, fx.Mask)
			}
			if r.Content != fx.Text {
				t.Errorf("content = %q, want %q", r.Content, fx.Text)
			}
			if r.Errors != 0 {
				t.Errorf("clean matrix needed %d corrections", r.Errors)
			}

			// Flip one module in every codeword-sized stripe of the first block
			// up to half the EC capacity: the decoder must still recover.
			blocks := BlockLayout(fx.Version, levelFromString(fx.Level))
			flips := blocks[0].EC / 2
			corrupted := make([]uint8, len(bits))
			copy(corrupted, bits)
			for i := 0; i < flips; i++ {
				p := l.dataPos[i*8*len(blocks)]
				corrupted[p[0]*n+p[1]] ^= 1
			}
			r2, err := DecodeMatrix(corrupted, fx.Version)
			if err != nil {
				t.Fatalf("DecodeMatrix with %d flipped modules: %v", flips, err)
			}
			if r2.Content != fx.Text {
				t.Errorf("corrupted content = %q, want %q", r2.Content, fx.Text)
			}
		})
	}
}

func TestParseSegments(t *testing.T) {
	tests := []struct {
		name    string
		bits    string
		version int
		want    string
		wantErr bool
	}{
		{"numeric 01234567 (ISO I.2)", "0001" + "0000001000" + "0000001100" + "0101011001" + "1000011" + "0000", 1, "01234567", false},
		{"alphanumeric AC-42 (ISO I.3)", "0010" + "000000101" + "00111001110" + "11100111001" + "000010" + "0000", 1, "AC-42", false},
		{"byte hi", "0100" + "00000010" + "01101000" + "01101001" + "0000", 1, "hi", false},
		{"eci utf8 then byte", "0111" + "00011010" + "0100" + "00000001" + "01100001" + "0000", 1, "a", false},
		{"kanji unsupported", "1000" + "00000001" + "0000000000000", 1, "", true},
		{"empty", "0000", 1, "", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data := bitsToBytes(tc.bits)
			payload, eci, err := parseSegments(data, tc.version)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got payload %q", payload)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseSegments: %v", err)
			}
			if got := payloadToString(payload, eci); got != tc.want {
				t.Errorf("payload = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestPayloadToStringLatin1(t *testing.T) {
	// 0xE1 0x6C is not valid UTF-8; without an ECI it must be read as Latin-1.
	got := payloadToString([]byte{'s', 'e', 'x', 'u', 0xE1, 'l'}, 0)
	if got != "sexuál" {
		t.Errorf("payloadToString = %q, want %q", got, "sexuál")
	}
	if got := payloadToString([]byte("caf\xc3\xa9"), 0); got != "café" {
		t.Errorf("valid UTF-8 altered: %q", got)
	}
}

func bitsToBytes(s string) []byte {
	for len(s)%8 != 0 {
		s += "0"
	}
	out := make([]byte, len(s)/8)
	for i, c := range s {
		if c == '1' {
			out[i/8] |= 1 << uint(7-i%8)
		}
	}
	return out
}
