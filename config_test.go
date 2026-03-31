package gqreader

import (
	"testing"

	"github.com/gemineo/gqreader/qrdecode"
)

func TestHarderConfig(t *testing.T) {
	cfg := HarderConfig()

	if !cfg.TryHarder {
		t.Error("HarderConfig: TryHarder should be true")
	}
	if len(cfg.FallbackDecoderFactories) != 1 {
		t.Errorf("HarderConfig: FallbackDecoderFactories len = %d, want 1", len(cfg.FallbackDecoderFactories))
	}
	if cfg.FallbackDecoderFactories[0] == nil {
		t.Error("HarderConfig: FallbackDecoderFactories[0] should not be nil")
	}
	if cfg.DecoderFactory == nil {
		t.Error("HarderConfig: DecoderFactory should not be nil")
	}
	// Verify the fallback factory is callable and returns an OpenCV decoder
	dec, err := cfg.FallbackDecoderFactories[0]()
	if err != nil {
		t.Fatalf("HarderConfig fallback factory error: %v", err)
	}
	defer dec.Close()
	if _, ok := dec.(*qrdecode.OpenCVDecoder); !ok {
		t.Errorf("HarderConfig fallback decoder type = %T, want *qrdecode.OpenCVDecoder", dec)
	}
}

func TestHarderConfig_ZBarConfigUnchanged(t *testing.T) {
	// Calling HarderConfig must not mutate ZBarConfig (value semantics check).
	base := ZBarConfig()
	_ = HarderConfig()
	after := ZBarConfig()

	if base.TryHarder != after.TryHarder {
		t.Error("ZBarConfig: TryHarder was mutated by HarderConfig call")
	}
	if len(base.FallbackDecoderFactories) != len(after.FallbackDecoderFactories) {
		t.Error("ZBarConfig: FallbackDecoderFactories was mutated by HarderConfig call")
	}
}

func TestWithFallbackDecoderFactories_ValueSemantics(t *testing.T) {
	factory := func() (qrdecode.Decoder, error) { return qrdecode.NewOpenCVDecoder() }
	original := ZBarConfig()
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

	cfg := ZBarConfig().
		WithFallbackDecoderFactories(f1).
		WithFallbackDecoderFactories(f2)

	if len(cfg.FallbackDecoderFactories) != 2 {
		t.Errorf("FallbackDecoderFactories len = %d, want 2 after two appends", len(cfg.FallbackDecoderFactories))
	}
}

func TestConfig_TryHarderDefaultFalse(t *testing.T) {
	for _, cfg := range []Config{DefaultConfig(), ZBarConfig(), MinimalConfig()} {
		if cfg.TryHarder {
			t.Errorf("config TryHarder should default to false")
		}
		if len(cfg.FallbackDecoderFactories) != 0 {
			t.Errorf("config FallbackDecoderFactories should default to empty")
		}
	}
}
