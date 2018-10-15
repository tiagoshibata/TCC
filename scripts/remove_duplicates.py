#!/usr/bin/env python3
'''Usage:
Removing duplicates:
find -not -empty -type f -printf "%s\n" | sort -rn | uniq -d | xargs -I{} -n1 find -type f -size {}c -print0 |
    xargs -0 md5sum | sort | uniq -w32 --all-repeated=separate | cut -c 35- | remove_duplicates.py
Downloading ImageNet images:
for f in ../imagenet_urls/* ; do cat $f | xargs -n 1 -d'\n' -P50 wget --trust-server-names -nv -T5 --tries=1 ; done
'''
from pathlib import Path
import sys


def remove_duplicates():
    lines = (x[:-1] for x in sys.stdin.readlines())
    while True:
        first = next(lines, None)
        if first is None:
            return
        while True:
            duplicate = next(lines, None)
            if not duplicate:
                break
            print('Duplicate: {}'.format(duplicate))
            Path(duplicate).unlink()

if __name__ == '__main__':
    remove_duplicates()
