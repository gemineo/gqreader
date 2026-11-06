package qrdecode

// ZBar decoder: wraps libzbar for fast QR code scanning.
//
// Compared to OpenCV QRCodeDetector, zbar uses a lighter-weight algorithm
// that is typically 3-5× faster on single-crop decoding.
// It also matches the decoder used by Python QReader (pyzbar wraps the same libzbar).
//
// Build requirements:
//   macOS: brew install zbar
//   Linux: apt-get install libzbar-dev   (Debian/Ubuntu)
//          dnf install zbar-devel        (Fedora/RHEL)

/*
#cgo pkg-config: zbar

#include <zbar.h>
#include <stdlib.h>

// zbarFourcc builds a FourCC code from four ASCII bytes.
// Equivalent to Python's ZBAR_FORMAT_Y800 / 'Y800'.
static unsigned long zbarFourcc(unsigned char a, unsigned char b,
                                unsigned char c, unsigned char d) {
    return ((unsigned long)a)        |
           ((unsigned long)b << 8)  |
           ((unsigned long)c << 16) |
           ((unsigned long)d << 24);
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"gocv.io/x/gocv"
)

// ZBarDecoder implements Decoder using the libzbar image scanner.
//
// ZBar is typically 3-5× faster than OpenCV QRCodeDetector on single-crop
// decoding and is the library used by Python QReader (via pyzbar).
//
// Each ZBarDecoder instance is NOT safe for concurrent use.
// The QReader pipeline creates a new instance per goroutine via DecoderFactory.
//
// # Known limitation: ISO-8859-1 byte sequences that form valid Shift-JIS
//
// When a QR code uses byte mode to store ISO-8859-1 (Latin-1) data, ZBar may
// misidentify the encoding if the raw bytes happen to form valid Shift-JIS
// double-byte sequences. For example, 'á' (U+00E1) is encoded as 0xE1 in
// Latin-1; if followed by a byte in [0x40–0xFC] such as 'l' (0x6C), ZBar
// treats the pair as a Shift-JIS character (獺, U+7B7A) rather than the
// correct Latin-1 value.
//
// OpenCV QRCodeDetector handles this correctly by converting the raw byte
// payload to UTF-8 using the QR code's declared encoding. Python QReader
// works around the ZBar bug by re-encoding the ZBar output through Big5
// ("encode as Big5, decode as UTF-8"), which coincidentally recovers the
// original Latin-1 value for some Western characters. This workaround is
// not principled and does not generalise.
//
// Callers that must decode QR codes containing ISO-8859-1 Central or Eastern
// European characters should use DefaultConfig() (ZBar+OpenCV fallback) instead
// of FastConfig() (ZBar only). This affects image QR2c in Dataset-QR-2020.
type ZBarDecoder struct {
	scanner *C.zbar_image_scanner_t
}

// NewZBarDecoder creates a ZBarDecoder configured for QR Code scanning only.
// Returns an error if libzbar is not available.
func NewZBarDecoder() (*ZBarDecoder, error) {
	scanner := C.zbar_image_scanner_create()
	if scanner == nil {
		return nil, fmt.Errorf("zbar_image_scanner_create failed")
	}

	// Disable all symbol types, then enable QR Code only.
	C.zbar_image_scanner_set_config(scanner, C.ZBAR_NONE, C.ZBAR_CFG_ENABLE, 0)
	C.zbar_image_scanner_set_config(scanner, C.ZBAR_QRCODE, C.ZBAR_CFG_ENABLE, 1)

	return &ZBarDecoder{scanner: scanner}, nil
}

// Decode scans img for QR codes using libzbar.
// The input may be color (BGR) or grayscale; color images are converted internally.
func (d *ZBarDecoder) Decode(img gocv.Mat) ([]DecodedQR, error) {
	if img.Empty() {
		return nil, nil
	}

	// ZBar requires Y800 (8-bit grayscale) input.
	var gray gocv.Mat
	var ownGray bool
	if img.Channels() == 1 {
		gray = img
	} else {
		gray = gocv.NewMat()
		ownGray = true
		gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
	}
	// gray is valid until we return; zbar image is destroyed before gray is closed.

	width := gray.Cols()
	height := gray.Rows()

	dataPtr, err := gray.DataPtrUint8()
	if err != nil {
		if ownGray {
			gray.Close()
		}
		return nil, fmt.Errorf("getting image data: %w", err)
	}

	// Build a zbar image borrowing the Mat's pixel buffer (no copy).
	zimg := C.zbar_image_create()
	C.zbar_image_set_format(zimg, C.zbarFourcc('Y', '8', '0', '0'))
	C.zbar_image_set_size(zimg, C.uint(width), C.uint(height))
	// NULL cleanup fn: zbar does not own or free the data pointer.
	C.zbar_image_set_data(zimg, unsafe.Pointer(&dataPtr[0]),
		C.ulong(width*height), nil)

	n := C.zbar_scan_image(d.scanner, zimg)

	var results []DecodedQR
	if n > 0 {
		sym := C.zbar_image_first_symbol(zimg)
		for sym != nil {
			if C.zbar_symbol_get_type(sym) == C.ZBAR_QRCODE {
				// Use the explicit length: payloads may contain NUL bytes,
				// which C.GoString would truncate at.
				n := C.zbar_symbol_get_data_length(sym)
				content := string(C.GoBytes(unsafe.Pointer(C.zbar_symbol_get_data(sym)), C.int(n)))
				if content != "" {
					results = append(results, DecodedQR{Content: content})
				}
			}
			sym = C.zbar_symbol_next(sym)
		}
	}

	// Destroy zbar image before releasing the gray Mat.
	C.zbar_image_destroy(zimg)
	if ownGray {
		gray.Close()
	}

	return results, nil
}

// Close releases the zbar image scanner.
func (d *ZBarDecoder) Close() error {
	if d.scanner != nil {
		C.zbar_image_scanner_destroy(d.scanner)
		d.scanner = nil
	}
	return nil
}
