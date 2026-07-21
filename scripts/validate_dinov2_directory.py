#!/usr/bin/env python3
"""Offline validation for a local Hugging Face DINOv2 model directory."""

import argparse

from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    print(type(model).__name__)
    print("parameters", sum(parameter.numel() for parameter in model.parameters()))
    print("hidden_size", model.config.hidden_size)
    print("layers", model.config.num_hidden_layers)
    print("patch_size", model.config.patch_size)


if __name__ == "__main__":
    main()
