#!/usr/bin/env python3
"""
Download the MNIST dataset and save each image as a 28x28 grayscale PNG.

Output structure:
    data/train/<label>/<index>.png
    data/test/<label>/<index>.png

Dependencies:
    pip install pypng
"""

import gzip
import os
import struct
import sys
from array import array
from urllib.request import urlretrieve

import png

# Use the stable S3 mirror that torchvision relies on.
URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download(url, dest):
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest):
        print(f"  already exists: {dest}")
        return
    print(f"  downloading: {os.path.basename(dest)} ...")
    urlretrieve(url, dest)


def read_idx_images(path):
    """Read an IDX image file (gzipped) and return (num_images, rows, cols, flat_data)."""
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic number {magic} for images file"
        data = array("B", f.read())
    return num_images, rows, cols, data


def read_idx_labels(path):
    """Read an IDX label file (gzipped) and return labels as an array."""
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic number {magic} for labels file"
        labels = array("b", f.read())
    return labels


def write_pngs(labels, data, num_images, rows, cols, output_dir):
    """Write each image as a greyscale PNG into output_dir/<label>/<index>.png."""
    # Pre-create label directories 0-9
    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)

    pixels_per_image = rows * cols
    for i in range(num_images):
        label = labels[i]
        out_path = os.path.join(output_dir, str(label), f"{i}.png")
        with open(out_path, "wb") as f:
            w = png.Writer(cols, rows, greyscale=True)
            img_rows = [
                data[i * pixels_per_image + r * cols : i * pixels_per_image + (r + 1) * cols]
                for r in range(rows)
            ]
            w.write(f, img_rows)

        if (i + 1) % 5000 == 0 or i == num_images - 1:
            print(f"    {i + 1}/{num_images}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Temporary download location for .gz files
    tmp_dir = os.path.join(DATA_DIR, "_raw")
    os.makedirs(tmp_dir, exist_ok=True)

    print("Downloading MNIST archives...")
    gz_paths = {}
    for key, url in URLS.items():
        filename = url.split("/")[-1]
        dest = os.path.join(tmp_dir, filename)
        download(url, dest)
        gz_paths[key] = dest

    for split, split_dir_name in [("train", "train"), ("test", "test")]:
        print(f"\nProcessing {split} set...")
        num_images, rows, cols, data = read_idx_images(gz_paths[f"{split}_images"])
        labels = read_idx_labels(gz_paths[f"{split}_labels"])
        assert len(labels) == num_images, "Mismatch between images and labels count"
        print(f"  {num_images} images, {rows}x{cols}")

        output_dir = os.path.join(DATA_DIR, split_dir_name)
        write_pngs(labels, data, num_images, rows, cols, output_dir)

    print(f"\nDone! PNGs saved to {DATA_DIR}/{{train,test}}/<label>/<index>.png")

    # Clean up raw .gz files
    print("Cleaning up raw downloads...")
    for path in gz_paths.values():
        os.remove(path)
    os.rmdir(tmp_dir)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
