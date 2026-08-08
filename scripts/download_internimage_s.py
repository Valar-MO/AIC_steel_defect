#!/usr/bin/env python3
"""Download the official InternImage-S classification checkpoint via Hugging Face Hub."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/weights/internimage_s"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download("OpenGVLab/InternImage", "internimage_s_1k_224.pth", local_dir=args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
