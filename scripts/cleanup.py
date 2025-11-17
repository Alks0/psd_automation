#!/usr/bin/env python3
"""Cleanup helpers for temporary PSD exports."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def cleanup_psd_tmp(tmp_root: Path, psd_name: str) -> None:
    target = tmp_root / psd_name
    if target.exists():
        shutil.rmtree(target)


def cleanup_stale(tmp_root: Path, hours: int) -> None:
    cutoff = datetime.now() - timedelta(hours=hours)
    for child in tmp_root.iterdir():
        if not child.is_dir():
            continue
        mtime = datetime.fromtimestamp(child.stat().st_mtime)
        if mtime < cutoff:
            shutil.rmtree(child)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    remove = sub.add_parser('remove', help='删除特定 PSD 的临时目录')
    remove.add_argument('tmp_root', type=Path)
    remove.add_argument('psd_name')

    stale = sub.add_parser('stale', help='清理过期临时目录')
    stale.add_argument('tmp_root', type=Path)
    stale.add_argument('--hours', type=int, default=24)

    args = parser.parse_args()
    if args.cmd == 'remove':
        cleanup_psd_tmp(args.tmp_root, args.psd_name)
    else:
        cleanup_stale(args.tmp_root, args.hours)


if __name__ == '__main__':
    main()
