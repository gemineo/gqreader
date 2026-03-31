package gqreader

import (
	"testing"

	"github.com/gemineo/gqreader/qrdecode"
)

func TestDefaultConfig_TryHarder(t *testing.T) {
	cfg := DefaultConfig()

	if !cfg.TryHarder {
		t.Error("DefaultConfig: TryHarder should be true")
	}
	if len(cfg.FallbackDecoderFactories) != 1 {
		t.Errorf("DefaultConfig: FallbackDecoderFactories len = %d, want 1", len(cfg.FallbackDecoderFactories))
	}
	if cfg.FallbackDecoderFactories[0] == nil {
		t.Error("DefaultConfig: FallbackDecoderFactories[0] should not be nil")
	}
	if cfg.DecoderFactory == nil {
		t.Error("DefaultConfig: DecoderFactory should not be nil")
	}
	// Verify the fallback factory produces an OpenCV decoder.
	dec, err := cfg.FallbackDecoderFactories[0]()
	if err != nil {
		t.Fatalf("DefaultConfig fallback factory error: %v", err)
	}
	defer dec.Close()
	if _, ok := dec.(*qrdecode.OpenCVDecoder); !ok {
		t.Errorf("DefaultConfig fallback decoder type = %T, want *qrdecode.OpenCVDecoder", dec)
	}
}

func TestFastConfig_NoFallback(t *testing.T) {
	cfg := FastConfig()

	if cfg.TryHarder {
		t.Error("FastConfig: TryHarder should be false")
	}
	if len(cfg.FallbackDecoderFactories) != 0 {
		t.Errorf("FastConfig: FallbackDecoderFactories len = %d, want 0", len(cfg.FallbackDecoderFactories))
	}
	if cfg.DecoderFactory == nil {
		t.Error("FastConfig: DecoderFactory should not be nil")
	}
}

func TestMinimalConfig_NoFallback(t *testing.T) {
	cfg := MinimalConfig()

	if cfg.TryHarder {
		t.Error("MinimalConfig: TryHarder should be false")
	}
	if len(cfg.FallbackDecoderFactories) != 0 {
		t.Errorf("MinimalConfig: FallbackDecoderFactories len = %d, want 0", len(cfg.FallbackDecoderFactories))
	}
}

func TestWithFallbackDecoderFactories_ValueSemantics(t *testing.T) {
	// FastConfig is used as a no-fallback baseline to verify value semantics.
	factory := func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() }
	original := FastConfig()
	extended := original.WithFallbackDecoderFactories(factory)

	if original.TryHarder {
		t.Error("original config TryHarder was mutated")
	}
	if len(original.FallbackDecoderFactories) != 0 {
		t.Errorf("original FallbackDecoderFactories mutated, len = %d", len(original.FallbackDecoderFactories))
	}
	if !extended.TryHarder {
		t.Error("extended config TryHarder should be true")
	}
	if len(extended.FallbackDecoderFactories) != 1 {
		t.Errorf("extended FallbackDecoderFactories len = %d, want 1", len(extended.FallbackDecoderFactories))
	}
}

func TestWithFallbackDecoderFactories_Append(t *testing.T) {
	f1 := func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() }
	f2 := func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() }

	cfg := FastConfig().
		WithFallbackDecoderFactories(f1).
		WithFallbackDecoderFactories(f2)

	if len(cfg.FallbackDecoderFactories) != 2 {
		t.Errorf("FallbackDecoderFactories len = %d, want 2 after two appends", len(cfg.FallbackDecoderFactories))
	}
}
