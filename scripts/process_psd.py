#!/usr/bin/env python3
"""Utilities for exporting PSD layers to PNG snippets and metadata."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from psd_tools import PSDImage
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "psd_tools is required. Install it via pip install psd-tools pillow."
    ) from exc

LOGGER = logging.getLogger("process_psd")


@dataclass
class LayerRecord:
    index: int
    name: str
    group_path: str
    visible: bool
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    opacity: float
    layer_id: Optional[int]
    layer_png: str


@dataclass
class ExportResult:
    psd: str
    composite_png: str
    layers: List[LayerRecord]
    tmp_dir: str


SAFE_REPLACEMENTS = {
    '/': '-',
    '\\': '-',
    '\0': '',
}


def _safe_name(name: str, fallback: str) -> str:
    name = name.strip() or fallback
    for bad, repl in SAFE_REPLACEMENTS.items():
        name = name.replace(bad, repl)
    return ''.join(ch if ch.isprintable() else '_' for ch in name)


def _fix_encoding(text: str) -> str:
    """修复 PSD 文件中的编码问题（Mac Roman -> GBK）"""
    if not text:
        return text
    try:
        # 尝试 Mac Roman -> GBK 转换（Live2D/Photoshop 常见问题）
        fixed = text.encode('macroman', errors='replace').decode('gbk', errors='replace')
        # 如果转换后的文本看起来更合理（没有太多替换字符），就使用它
        if fixed.count('?') < len(fixed) * 0.3:  # 如果替换字符少于30%
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError, LookupError):
        pass
    return text


def _group_path(layer) -> str:
    segments: List[str] = []
    parent = getattr(layer, 'parent', None)
    while parent is not None and getattr(parent, 'name', None) is not None:
        if parent.name:
            segments.append(_fix_encoding(parent.name))
        parent = getattr(parent, 'parent', None)
    return '/'.join(reversed(segments))


def _iter_layers(layers: Iterable, include_hidden: bool) -> Iterable:
    for layer in layers:
        if layer.is_group():
            yield from _iter_layers(layer, include_hidden)
            continue
        if not include_hidden and not layer.visible:
            continue
        yield layer


def export_psd(psd_path: Path, tmp_root: Path, include_hidden: bool = False) -> ExportResult:
    psd = PSDImage.open(psd_path)
    psd_name = psd_path.stem
    tmp_dir = tmp_root / psd_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    composite_path = tmp_dir / f"{psd_name}_composite.png"
    if not composite_path.exists():
        psd.composite().save(composite_path)

    layers: List[LayerRecord] = []
    for idx, layer in enumerate(_iter_layers(psd, include_hidden), start=1):
        bbox = layer.bbox
        if hasattr(bbox, "width"):
            left, top, right, bottom = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            width, height = bbox.width, bbox.height
        else:
            left, top, right, bottom = bbox
            width = max(0, right - left)
            height = max(0, bottom - top)
        if width == 0 or height == 0:
            LOGGER.debug("Skipping empty layer %s", layer.name)
            continue
        try:
            pil_image = layer.topil()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to rasterize layer %s: %s", layer.name, exc)
            continue
        safe = _safe_name(layer.name or f"layer_{idx:04d}", f"layer_{idx:04d}")
        png_path = tmp_dir / f"layer_{idx:04d}_{safe}.png"
        pil_image.save(png_path)
        group_path = _group_path(layer)
        record = LayerRecord(
            index=idx,
            name=layer.name or safe,
            group_path=group_path,
            visible=layer.visible,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            width=width,
            height=height,
            opacity=layer.opacity,
            layer_id=getattr(layer, 'layer_id', None),
            layer_png=str(png_path),
        )
        layers.append(record)

    return ExportResult(
        psd=psd_path.name,
        composite_png=str(composite_path),
        layers=layers,
        tmp_dir=str(tmp_dir),
    )


def _write_metadata(result: ExportResult, output: Path) -> None:
    payload = {
        'psd': result.psd,
        'composite': result.composite_png,
        'layers': [asdict(layer) for layer in result.layers],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('psd', type=Path, help='PSD 路径')
    parser.add_argument('--tmp', type=Path, default=Path('tmp'))
    parser.add_argument('--include-hidden', action='store_true')
    parser.add_argument('--metadata', type=Path, help='导出 JSON 元数据路径')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    result = export_psd(args.psd, args.tmp, include_hidden=args.include_hidden)
    LOGGER.info('Composite: %s', result.composite_png)
    LOGGER.info('Exported %d layers', len(result.layers))
    if args.metadata:
        _write_metadata(result, args.metadata)
        LOGGER.info('Metadata saved to %s', args.metadata)


if __name__ == '__main__':
    main()
