"""Download FER-2013 into data/ as two Parquet files.

FER-2013 (Goodfellow et al., ICML 2013 representation learning challenge):
35,887 grayscale 48x48 face images labelled with 7 emotions.
Mirror: https://huggingface.co/datasets/Jeneral/fer-2013
Each row holds PNG bytes (img_bytes) and a class index (labels).
"""

from pathlib import Path
import urllib.request

URL = "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
DATA = Path(__file__).parent / "data"


def main():
    DATA.mkdir(exist_ok=True)
    for split in ("train", "test"):
        target = DATA / f"{split}.parquet"
        if target.exists():
            print(f"{split}: already downloaded")
            continue
        print(f"{split}: downloading...")
        urllib.request.urlretrieve(URL.format(split=split), target)
        print(f"{split}: {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
