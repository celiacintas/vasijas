#!/usr/bin/env python3
"""Script to download folders from Google Drive."""

import sys
import os
import subprocess


def download_folder(url_or_id: str, output_dir: str = ".", cookie_file: str = None) -> None:
    """Download a folder from Google Drive.

    Args:
        url_or_id: Google Drive folder URL or folder ID
        output_dir: Local directory to save files to
        cookie_file: Path to cookies.txt for authenticated downloads
    """
    cmd = ["gdown", "--folder", "--output", output_dir, url_or_id]
    if cookie_file and os.path.exists(cookie_file):
        cmd.extend(["--cookies", cookie_file])
    subprocess.run(cmd, check=True)
    print(f"Downloaded to: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_folder.py <folder_url_or_id> [output_dir] [cookie_file]")
        sys.exit(1)

    url_or_id = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    cookie_file = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        download_folder(url_or_id, output_dir, cookie_file)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
