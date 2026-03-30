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
// The pipeline matches Python QReader's brute-force approach:
//
//	For each crop (bbox, perspective):
//	  1. Try direct decode (no preprocessing)
//	  2. For each scale factor:
//	     Try each flavor (identity, inversion, gray+otsu+blur, sharpen+gray+blur)
//	     Return on first success
func decodeDetection(
	img gocv.Mat,
	det qrdet.Detection,
	decoder qrdecode.Decoder,
	config Config,
) QRResult {
	result := QRResult{Detection: det}

	// Build list of crops to try
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

	flavors := imgprep.QRDecodeStrategies()

	for _, crop := range crops {
		// 1. Try direct decode on crop (no preprocessing, no scaling)
		if content, strategy := tryDecode(decoder, crop.mat, crop.name+"/direct"); content != "" {
			result.Content = content
			result.Decoded = true
			result.Strategy = strategy
			return result
		}

		// 2. Try each scale factor
		scales := config.ScaleFactors
		if len(scales) == 0 {
			scales = []float64{1.0}
		}

		for _, scale := range scales {
			scaled := scaleImage(crop.mat, scale)
			if scaled.Empty() {
				continue
			}

			// Try each preprocessing flavor on the scaled image
			for _, flavor := range flavors {
				processed, err := flavor.Apply(scaled)
				if err != nil {
					continue
				}

				strategyName := crop.name + "/scale_" + scaleLabel(scale) + "/" + flavor.Name()
				if content, _ := tryDecode(decoder, processed, strategyName); content != "" {
					processed.Close()
					scaled.Close()
					result.Content = content
					result.Decoded = true
					result.Strategy = strategyName
					return result
				}
				processed.Close()
			}

			scaled.Close()
		}
	}

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
