#!/usr/bin/env python3
"""Script to download folders from Google Drive."""

import sys
import gdown


def download_folder(url_or_id: str, output_dir: str = ".") -> None:
    """Download a folder from Google Drive.

    Args:
        url_or_id: Google Drive folder URL or folder ID
        output_dir: Local directory to save files to
    """
    gdown.download_folder(url=url_or_id, output=output_dir)
    print(f"Downloaded to: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_folder.py <folder_url_or_id> [output_dir]")
        sys.exit(1)

    url_or_id = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    try:
        download_folder(url_or_id, output_dir)
    except gdown.DownloadError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()