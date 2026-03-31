//go:build wechat

package qrdecode

import (
	"fmt"

	"gocv.io/x/gocv"
	"gocv.io/x/gocv/contrib"
)

// WeChatDecoder decodes QR codes using the WeChat QRCode detector from OpenCV contrib.
//
// It bundles its own CNN-based QR detection and a super-resolution model, making it
// effective for small or low-contrast QR codes that standard decoders fail on.
//
// Requires four Caffe model files (download with `task wechat-models`):
//   - detect.prototxt + detect.caffemodel  — detection model
//   - sr.prototxt + sr.caffemodel          — super-resolution model
type WeChatDecoder struct {
	qr *contrib.WeChatQRCode
}

// NewWeChatDecoder creates a WeChatDecoder from the given Caffe model file paths.
// Returns an error if the model files cannot be loaded.
func NewWeChatDecoder(detectProto, detectModel, srProto, srModel string) (*WeChatDecoder, error) {
	wq := contrib.NewWeChatQRCode(detectProto, detectModel, srProto, srModel)
	if wq == nil {
		return nil, fmt.Errorf("wechat qrcode: failed to load models (check file paths)")
	}
	return &WeChatDecoder{qr: wq}, nil
}

// Decode attempts to detect and decode QR codes from the given image.
// Returns all decoded QR codes found (WeChatQRCode supports multi-QR).
func (d *WeChatDecoder) Decode(img gocv.Mat) ([]DecodedQR, error) {
	if img.Empty() {
		return nil, nil
	}

	points := make([]gocv.Mat, 0)
	decoded := d.qr.DetectAndDecode(img, &points)
	for _, m := range points {
		m.Close()
	}

	if len(decoded) == 0 {
		return nil, nil
	}

	results := make([]DecodedQR, 0, len(decoded))
	for _, s := range decoded {
		if s != "" {
			results = append(results, DecodedQR{Content: s})
		}
	}
	return results, nil
}

// Close is a no-op. contrib.WeChatQRCode does not expose a destructor.
func (d *WeChatDecoder) Close() error { return nil }

// Ensure WeChatDecoder implements Decoder.
var _ Decoder = (*WeChatDecoder)(nil)
