package gqreader

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"

	"github.com/gemineo/gqreader/imgprep"
	"github.com/gemineo/gqreader/qrdecode"
	"github.com/gemineo/gqreader/qrdet"
)

// decodeDetection attempts to decode a single detected QR code by trying
// multiple crop methods, scale factors, and preprocessing flavors.
//
// The pipeline runs in two phases to keep average latency low:
//
//	Phase 1 — primary strategies (fast, covers most real-world QR codes):
//	  For each crop (bbox, then perspective if enabled):
//	    1. Direct decode (no preprocessing, no scaling)
//	    2. For each scale factor: try each primary flavor
//
//	Phase 2 — fallback strategies (only when Phase 1 yields nothing):
//	  For each crop × scale factor: try each fallback flavor
func decodeDetection(
	img gocv.Mat,
	det qrdet.Detection,
	decoder qrdecode.Decoder,
	config Config,
) QRResult {
	result := QRResult{Detection: det}

	// Build list of crops to try.
	type namedCrop struct {
		name string
		mat  gocv.Mat
	}
	crops := make([]namedCrop, 0, 2)

	bboxCrop := cropBBox(img, det)
	if !bboxCrop.Empty() {
		crops = append(crops, namedCrop{name: "bbox", mat: bboxCrop})
	}

	if config.TryPerspective {
		perspCrop := cropPerspective(img, det)
		if !perspCrop.Empty() {
			crops = append(crops, namedCrop{name: "perspective", mat: perspCrop})
		}
	}

	defer func() {
		for _, c := range crops {
			c.mat.Close()
		}
	}()

	scales := config.ScaleFactors
	if len(scales) == 0 {
		scales = []float64{1.0}
	}

	// attempt tries all crops × scales × flavors and updates result on first success.
	// When includeDirect is true, each crop is also tried as-is (no preprocessing, no
	// scaling) before the scale loop — this covers the common easy case cheaply.
	attempt := func(flavors []imgprep.Strategy, includeDirect bool) bool {
		for _, crop := range crops {
			if includeDirect {
				if content, s := tryDecode(decoder, crop.mat, crop.name+"/direct"); content != "" {
					result.Content = content
					result.Decoded = true
					result.Strategy = s
					return true
				}
			}

			for _, scale := range scales {
				scaled := scaleImage(crop.mat, scale)
				if scaled.Empty() {
					continue
				}

				for _, flavor := range flavors {
					processed, err := flavor.Apply(scaled)
					if err != nil {
						continue
					}
					sName := crop.name + "/scale_" + scaleLabel(scale) + "/" + flavor.Name()
					if content, _ := tryDecode(decoder, processed, sName); content != "" {
						processed.Close()
						scaled.Close()
						result.Content = content
						result.Decoded = true
						result.Strategy = sName
						return true
					}
					processed.Close()
				}

				scaled.Close()
			}
		}
		return false
	}

	// Phase 1: primary strategies (direct decode included).
	if attempt(imgprep.QRDecodeStrategies(), true) {
		return result
	}

	// Phase 2: fallback strategies, only reached when Phase 1 exhausts all options.
	attempt(imgprep.QRDecodeFallbackStrategies(), false)
	return result
}

// tryDecode attempts to decode a QR code from the given image.
// Returns the decoded content and strategy name, or empty string if decoding failed.
func tryDecode(decoder qrdecode.Decoder, img gocv.Mat, strategy string) (string, string) {
	if img.Empty() {
		return "", ""
	}

	results, err := decoder.Decode(img)
	if err != nil || len(results) == 0 {
		return "", ""
	}

	return results[0].Content, strategy
}

// scaleImage resizes an image by the given factor.
// Returns an empty Mat if the result would be too small or too large.
func scaleImage(src gocv.Mat, factor float64) gocv.Mat {
	if factor == 1.0 {
		dst := gocv.NewMat()
		src.CopyTo(&dst)
		return dst
	}

	newW := int(float64(src.Cols()) * factor)
	newH := int(float64(src.Rows()) * factor)

	const minDim = 25
	const maxDim = 4096

	if newW < minDim || newH < minDim || newW > maxDim || newH > maxDim {
		return gocv.NewMat()
	}

	dst := gocv.NewMat()
	gocv.Resize(src, &dst, image.Pt(newW, newH), 0, 0, gocv.InterpolationCubic)
	return dst
}

// scaleLabel formats a scale factor as a string for strategy naming.
func scaleLabel(factor float64) string {
	if factor == 1.0 {
		return "1.00"
	}
	return fmt.Sprintf("%.2f", factor)
}
