package qrstruct

import (
	"bytes"
	"math/rand"
	"testing"
)

// rsEncode appends nsym Reed-Solomon codewords to data (test helper).
func rsEncode(data []byte, nsym int) []byte {
	gen := []byte{1}
	for i := 0; i < nsym; i++ {
		gen = polyMul(gen, []byte{1, gfPow2(i)})
	}
	msg := make([]byte, len(data)+nsym)
	copy(msg, data)
	for i := 0; i < len(data); i++ {
		coef := msg[i]
		if coef == 0 {
			continue
		}
		for j := 1; j < len(gen); j++ {
			msg[i+j] ^= gfMul(gen[j], coef)
		}
	}
	copy(msg, data)
	return msg
}

func TestRSEncodeKnownVector(t *testing.T) {
	// "HELLO WORLD" in alphanumeric mode, version 1-M (ISO 18004 Annex I.3).
	data := []byte{0x20, 0x5B, 0x0B, 0x78, 0xD1, 0x72, 0xDC, 0x4D, 0x43, 0x40, 0xEC, 0x11, 0xEC, 0x11, 0xEC, 0x11}
	want := []byte{0xC4, 0x23, 0x27, 0x77, 0xEB, 0xD7, 0xE7, 0xE2, 0x5D, 0x17}
	got := rsEncode(data, 10)[16:]
	if !bytes.Equal(got, want) {
		t.Fatalf("EC codewords = % X, want % X", got, want)
	}
}

func TestRSDecodeRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	tests := []struct {
		name     string
		dataLen  int
		nsym     int
		errors   int
		erasures int
		wantOK   bool
	}{
		{"clean", 19, 7, 0, 0, true},
		{"max errors v1-L", 19, 7, 3, 0, true},
		{"max errors v2-Q", 22, 22, 11, 0, true},
		{"errors+erasures within bound", 28, 16, 4, 8, true},
		{"erasures only at bound", 34, 10, 0, 10, true},
		{"one error too many", 19, 7, 4, 0, false},
		{"long block", 116, 30, 15, 0, true},
		{"erasure bound exceeded", 16, 10, 0, 11, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for trial := 0; trial < 20; trial++ {
				data := make([]byte, tc.dataLen)
				rng.Read(data)
				msg := rsEncode(data, tc.nsym)
				corrupted := make([]byte, len(msg))
				copy(corrupted, msg)
				perm := rng.Perm(len(msg))
				var erase []int
				for i := 0; i < tc.erasures && i < len(perm); i++ {
					corrupted[perm[i]] ^= byte(rng.Intn(255) + 1)
					erase = append(erase, perm[i])
				}
				for i := tc.erasures; i < tc.erasures+tc.errors && i < len(perm); i++ {
					corrupted[perm[i]] ^= byte(rng.Intn(255) + 1)
				}
				n, err := RSDecode(corrupted, tc.nsym, erase)
				if !tc.wantOK {
					if err == nil && bytes.Equal(corrupted[:tc.dataLen], data) {
						// Correcting beyond the bound is not guaranteed to fail, but
						// a silent wrong answer is: the data must not be claimed right.
						continue
					}
					if err == nil {
						t.Fatalf("trial %d: decode beyond capacity returned wrong data without error", trial)
					}
					continue
				}
				if err != nil {
					t.Fatalf("trial %d: RSDecode error: %v", trial, err)
				}
				if !bytes.Equal(corrupted[:tc.dataLen], data) {
					t.Fatalf("trial %d: data mismatch after correction", trial)
				}
				if n != tc.errors+tc.erasures {
					t.Errorf("trial %d: corrected %d errata, want %d", trial, n, tc.errors+tc.erasures)
				}
			}
		})
	}
}
