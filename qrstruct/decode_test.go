package qrstruct

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type warpedFixture struct {
	File      string        `json:"file"`
	Text      string        `json:"text"`
	Version   int           `json:"version"`
	Level     string        `json:"level"`
	Quad      [4][2]float64 `json:"quad"`
	ExactQuad [4][2]float64 `json:"exact_quad"`
}

func quadFromArray(a [4][2]float64) Quad {
	var q Quad
	for i := range a {
		q[i] = Point{a[i][0], a[i][1]}
	}
	return q
}

// TestDecodeWarpedFixtures runs the full decoder on rendered symbols with
// perspective, rotation, cylinder bending, blur and noise, starting from a
// jittered quad as a detector would provide.
func TestDecodeWarpedFixtures(t *testing.T) {
	dir := filepath.Join("testdata", "warped")
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatalf("reading manifest: %v", err)
	}
	var fixtures []warpedFixture
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatalf("parsing manifest: %v", err)
	}
	for _, fx := range fixtures {
		t.Run(fx.File, func(t *testing.T) {
			img := loadGray(t, filepath.Join(dir, fx.File))
			start := time.Now()
			r, err := Decode(img, quadFromArray(fx.Quad), DefaultOptions())
			elapsed := time.Since(start)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if r.Content != fx.Text {
				t.Errorf("content = %q, want %q", r.Content, fx.Text)
			}
			if r.Version != fx.Version || r.Level != levelFromString(fx.Level) {
				t.Errorf("symbol = v%d-%s, want v%d-%s", r.Version, r.Level, fx.Version, fx.Level)
			}
			t.Logf("%s: v%d-%s mask %d, %s fit, %d errors, %d erasures, score %.0f, %v",
				fx.File, r.Version, r.Level, r.Mask, r.Stage, r.Errors, r.Erasures, r.Score, elapsed.Round(time.Millisecond))
		})
	}
}

func TestDecodeRejectsGarbage(t *testing.T) {
	img := loadGray(t, filepath.Join("testdata", "warped", "flat_v2L.png"))
	// A quad over the quiet zone only: nothing to decode.
	q := Quad{{2, 2}, {20, 2}, {20, 20}, {2, 20}}
	if r, err := Decode(img, q, DefaultOptions()); err == nil {
		t.Fatalf("expected an error on a blank region, got %q", r.Content)
	}
}

func TestOptionsNormalise(t *testing.T) {
	o := Options{}.normalise()
	d := DefaultOptions()
	if o.MinVersion != d.MinVersion || o.MaxVersion != d.MaxVersion || o.MaxCandidates != d.MaxCandidates || o.Workers != d.Workers {
		t.Errorf("zero Options normalised to %+v, want defaults %+v", o, d)
	}
	o = Options{MinVersion: 5, MaxVersion: 2}.normalise()
	if o.MaxVersion < o.MinVersion {
		t.Errorf("MaxVersion %d below MinVersion %d after normalisation", o.MaxVersion, o.MinVersion)
	}
}

func BenchmarkDecodeCylinder(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "warped", "manifest.json"))
	if err != nil {
		b.Fatal(err)
	}
	var fixtures []warpedFixture
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		b.Fatal(err)
	}
	var fx warpedFixture
	for _, f := range fixtures {
		if f.File == "cylinder_v2Q.png" {
			fx = f
		}
	}
	f, err := os.Open(filepath.Join("testdata", "warped", fx.File))
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	t := &testing.T{}
	img := loadGray(t, filepath.Join("testdata", "warped", fx.File))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Decode(img, quadFromArray(fx.Quad), DefaultOptions()); err != nil {
			b.Fatal(err)
		}
	}
}
