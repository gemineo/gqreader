package qrstruct

import (
	"errors"
	"fmt"
	"unicode/utf8"
)

// Segment modes (ISO 18004 §7.4.1).
const (
	modeTerminator       = 0
	modeNumeric          = 1
	modeAlphanumeric     = 2
	modeStructuredAppend = 3
	modeByte             = 4
	modeFNC1First        = 5
	modeECI              = 7
	modeKanji            = 8
	modeFNC1Second       = 9
)

// ECI assignment numbers that influence text conversion.
const (
	eciISO88591 = 3
	eciUTF8     = 26
)

const alphanumericChars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

// bitReader reads big-endian bit fields from a byte slice.
type bitReader struct {
	data []byte
	pos  int // bit position
}

func (b *bitReader) remaining() int { return len(b.data)*8 - b.pos }

func (b *bitReader) read(n int) (int, bool) {
	if n > b.remaining() {
		return 0, false
	}
	v := 0
	for i := 0; i < n; i++ {
		byteIdx := b.pos >> 3
		bit := (b.data[byteIdx] >> uint(7-(b.pos&7))) & 1
		v = v<<1 | int(bit)
		b.pos++
	}
	return v, true
}

// charCountBits returns the length of the character count field for a mode
// in the given version.
func charCountBits(mode, version int) int {
	group := 0
	switch {
	case version >= 27:
		group = 2
	case version >= 10:
		group = 1
	}
	switch mode {
	case modeNumeric:
		return [3]int{10, 12, 14}[group]
	case modeAlphanumeric:
		return [3]int{9, 11, 13}[group]
	case modeByte:
		return [3]int{8, 16, 16}[group]
	case modeKanji:
		return [3]int{8, 10, 12}[group]
	}
	return 0
}

// errUnsupportedMode is returned for Kanji segments, which this decoder does
// not convert.
var errUnsupportedMode = errors.New("unsupported segment mode")

// parseSegments decodes the data codewords into the payload bytes. It returns
// the bytes, the last ECI designator seen (0 if none) and an error when the
// stream is malformed.
func parseSegments(data []byte, version int) ([]byte, int, error) {
	br := &bitReader{data: data}
	var out []byte
	eci := 0
	for br.remaining() >= 4 {
		mode, _ := br.read(4)
		if mode == modeTerminator {
			break
		}
		switch mode {
		case modeNumeric:
			cnt, ok := br.read(charCountBits(mode, version))
			if !ok {
				return nil, eci, errors.New("truncated numeric count")
			}
			for cnt >= 3 {
				v, ok := br.read(10)
				if !ok || v > 999 {
					return nil, eci, errors.New("bad numeric triple")
				}
				out = append(out, byte('0'+v/100), byte('0'+(v/10)%10), byte('0'+v%10))
				cnt -= 3
			}
			if cnt == 2 {
				v, ok := br.read(7)
				if !ok || v > 99 {
					return nil, eci, errors.New("bad numeric pair")
				}
				out = append(out, byte('0'+v/10), byte('0'+v%10))
			} else if cnt == 1 {
				v, ok := br.read(4)
				if !ok || v > 9 {
					return nil, eci, errors.New("bad numeric digit")
				}
				out = append(out, byte('0'+v))
			}
		case modeAlphanumeric:
			cnt, ok := br.read(charCountBits(mode, version))
			if !ok {
				return nil, eci, errors.New("truncated alphanumeric count")
			}
			for cnt >= 2 {
				v, ok := br.read(11)
				if !ok || v >= 45*45 {
					return nil, eci, errors.New("bad alphanumeric pair")
				}
				out = append(out, alphanumericChars[v/45], alphanumericChars[v%45])
				cnt -= 2
			}
			if cnt == 1 {
				v, ok := br.read(6)
				if !ok || v >= 45 {
					return nil, eci, errors.New("bad alphanumeric char")
				}
				out = append(out, alphanumericChars[v])
			}
		case modeByte:
			cnt, ok := br.read(charCountBits(mode, version))
			if !ok || cnt*8 > br.remaining() {
				return nil, eci, errors.New("truncated byte segment")
			}
			for i := 0; i < cnt; i++ {
				v, _ := br.read(8)
				out = append(out, byte(v))
			}
		case modeECI:
			first, ok := br.read(8)
			if !ok {
				return nil, eci, errors.New("truncated ECI")
			}
			switch {
			case first&0x80 == 0:
				eci = first & 0x7f
			case first&0xc0 == 0x80:
				second, ok := br.read(8)
				if !ok {
					return nil, eci, errors.New("truncated ECI")
				}
				eci = (first&0x3f)<<8 | second
			default:
				rest, ok := br.read(16)
				if !ok {
					return nil, eci, errors.New("truncated ECI")
				}
				eci = (first&0x1f)<<16 | rest
			}
		case modeStructuredAppend:
			if _, ok := br.read(16); !ok {
				return nil, eci, errors.New("truncated structured append header")
			}
		case modeFNC1First:
			// No payload; GS1 framing is left to the caller.
		case modeFNC1Second:
			if _, ok := br.read(8); !ok {
				return nil, eci, errors.New("truncated FNC1 indicator")
			}
		case modeKanji:
			return nil, eci, fmt.Errorf("%w: kanji", errUnsupportedMode)
		default:
			return nil, eci, fmt.Errorf("%w: mode %d", errUnsupportedMode, mode)
		}
	}
	if len(out) == 0 {
		return nil, eci, errors.New("empty payload")
	}
	return out, eci, nil
}

// payloadToString converts the payload bytes to a Go string following the
// ECI designator when present, and otherwise treating valid UTF-8 as UTF-8
// and anything else as ISO-8859-1.
func payloadToString(payload []byte, eci int) string {
	switch {
	case eci == eciUTF8:
		return string(payload)
	case eci == eciISO88591:
		return latin1ToUTF8(payload)
	case eci == 0 && utf8.Valid(payload):
		return string(payload)
	case eci == 0:
		return latin1ToUTF8(payload)
	default:
		return string(payload)
	}
}

func latin1ToUTF8(b []byte) string {
	out := make([]rune, len(b))
	for i, c := range b {
		out[i] = rune(c)
	}
	return string(out)
}
