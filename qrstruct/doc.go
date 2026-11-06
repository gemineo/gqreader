// Package qrstruct implements a structured QR code decoder.
//
// Classical decoders (ZBar, OpenCV, ZXing) locate the finder patterns, derive a
// sampling grid from a handful of anchor points and then read the modules. On
// low-resolution or non-planar symbols (3-4 px per module on a cup or a bottle)
// that grid is typically off by a fraction of a module, which corrupts more
// codewords than Reed-Solomon can repair even though the symbol was located.
//
// This package takes the opposite route. Given an approximate quadrilateral
// around the symbol (for example the YOLO detection quad), it:
//
//  1. parametrises the sampling grid (a homography from the module lattice to
//     the image, optionally composed with a tilted-axis cylinder model);
//  2. fits the grid parameters by maximising agreement on the modules whose
//     value is known a priori for a hypothesised version: finder patterns,
//     separators, timing patterns, alignment patterns, the dark module, the
//     BCH-protected format information and, for versions 7+, the version
//     information. Roughly a third of the symbol is known before any decoding;
//  3. samples the module centres, binarises, reads the format information,
//     unmasks, de-interleaves the codewords and runs Reed-Solomon error
//     correction with progressive erasures on the least reliable codewords.
//
// The version and orientation are unknown on entry and are identified by
// running the fit for every candidate and keeping the best-scoring ones.
//
// The package is pure Go (no CGO) and works on *image.Gray.
package qrstruct
