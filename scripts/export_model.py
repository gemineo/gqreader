#!/usr/bin/env python3
"""
Export QRdet YOLOv8 segmentation model to ONNX format.

This script:
1. Downloads the QRdet model weights from GitHub releases
2. Loads the model using Ultralytics
3. Exports to ONNX with optimized settings for CPU inference

Usage:
    python export_model.py [--size s] [--output ../models/qrdet-s.onnx]

Requirements:
    pip install ultralytics onnx onnxsim
"""

import argparse
import os
import sys
from pathlib import Path

# Model download URLs from QRdet releases
MODEL_URLS = {
    "n": "https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release/qrdet-n.pt",
    "s": "https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release/qrdet-s.pt",
    "m": "https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release/qrdet-m.pt",
    "l": "https://github.com/Eric-Canas/qrdet/releases/download/v2.0_release/qrdet-l.pt",
}


def download_model(size: str, output_dir: Path) -> Path:
    """Download QRdet model weights if not already present."""
    import requests
    from tqdm import tqdm

    url = MODEL_URLS.get(size)
    if not url:
        raise ValueError(f"Unknown model size: {size}. Choose from: {list(MODEL_URLS.keys())}")

    output_path = output_dir / f"qrdet-{size}.pt"

    if output_path.exists():
        print(f"Model already exists: {output_path}")
        return output_path

    print(f"Downloading QRdet-{size} model from {url}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"qrdet-{size}.pt") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Downloaded to: {output_path}")
    return output_path


def export_to_onnx(pt_path: Path, onnx_path: Path, input_size: int = 640) -> Path:
    """
    Export PyTorch model to ONNX format.

    Export parameters are chosen for:
    - CPU inference compatibility (opset 12)
    - Fixed input size (640x640) for simpler tensor handling
    - Model simplification for faster inference (if onnxsim available)
    """
    from ultralytics import YOLO

    print(f"Loading model from: {pt_path}")
    model = YOLO(str(pt_path))

    print(f"Exporting to ONNX (input size: {input_size}x{input_size})...")

    # Export with settings optimized for CPU inference
    # Reference: https://docs.ultralytics.com/modes/export/
    # Note: simplify=True requires onnxsim which needs cmake
    # We skip it and let the user install onnxsim manually if needed
    export_path = model.export(
        format="onnx",
        imgsz=input_size,      # Fixed input size
        opset=12,              # Good CPU compatibility
        simplify=False,        # Skip simplify (requires cmake/onnxsim)
        dynamic=False,         # Fixed batch size (simpler Go integration)
        half=False,            # Full precision (FP32) for CPU
    )

    # Move to desired output location if different
    exported = Path(export_path)
    if exported != onnx_path:
        import shutil
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(onnx_path))
        print(f"Moved to: {onnx_path}")
    else:
        print(f"Exported to: {onnx_path}")

    return onnx_path


def verify_onnx(onnx_path: Path) -> None:
    """Verify the exported ONNX model."""
    import onnx

    print(f"Verifying ONNX model: {onnx_path}")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    print("\nModel inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    print("\nModel outputs:")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    # Get file size
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\nModel size: {size_mb:.2f} MB")
    print("ONNX model is valid!")


def main():
    parser = argparse.ArgumentParser(
        description="Export QRdet YOLOv8 model to ONNX format"
    )
    parser.add_argument(
        "--size",
        choices=["n", "s", "m", "l"],
        default="s",
        help="Model size: n(ano), s(mall), m(edium), l(arge). Default: s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path. Default: ../models/qrdet-{size}.onnx",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Model input size. Default: 640",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use existing .pt file",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX verification step",
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"

    if args.output:
        onnx_path = args.output
    else:
        onnx_path = models_dir / f"qrdet-{args.size}.onnx"

    pt_path = models_dir / f"qrdet-{args.size}.pt"

    # Download model if needed
    if not args.skip_download:
        pt_path = download_model(args.size, models_dir)
    elif not pt_path.exists():
        print(f"Error: Model not found: {pt_path}", file=sys.stderr)
        print("Run without --skip-download to download the model.", file=sys.stderr)
        sys.exit(1)

    # Export to ONNX
    export_to_onnx(pt_path, onnx_path, args.input_size)

    # Verify the model
    if not args.skip_verify:
        verify_onnx(onnx_path)

    print(f"\nSuccess! ONNX model ready at: {onnx_path}")


if __name__ == "__main__":
    main()
