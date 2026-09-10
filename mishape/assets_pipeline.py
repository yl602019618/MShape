"""Read MiShape's immutable, preprocessed polygon asset library.

Blender conversion is an optional offline operation; runtime use requires only the
standard library. See ``scripts/prepare_assets.py`` for provenance and transforms.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

ASSET_ROOT = Path(__file__).resolve().parent / 'assets'


def load_catalog() -> list[dict[str, Any]]:
    """Return the local vehicle catalog, preserving provenance and calibration."""
    path = ASSET_ROOT / 'catalog.json'
    if not path.exists():
        return []
    content = json.loads(path.read_text())
    return content.get('assets', []) if isinstance(content, dict) else content


def asset_entry(asset_id: str) -> dict[str, Any]:
    for entry in load_catalog():
        if entry['id'] == asset_id:
            return entry
    raise ValueError(f'Unknown asset: {asset_id}')


def load_asset(asset_id: str, *, full: bool = False) -> dict[str, Any]:
    """Load viewport or full polygon data using only a catalog-approved path."""
    entry = asset_entry(asset_id)
    key = 'full_mesh_path' if full else 'path'
    path = (ASSET_ROOT / entry[key]).resolve()
    if not path.is_relative_to(ASSET_ROOT.resolve()):
        raise ValueError('Asset path is outside the local library')
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def validate_asset(model: dict[str, Any]) -> dict[str, int]:
    """Validate the polygon/triangle indexing contract, raising on corruption."""
    import math
    vertices = model['vertices']
    faces = model['faces']
    if len(vertices) % 3 or len(faces) % 3:
        raise ValueError('Coordinate and triangle buffers must be multiples of 3')
    count = len(vertices) // 3
    if count == 0 or not all(math.isfinite(v) for v in vertices):
        raise ValueError('Asset contains empty or non-finite geometry')
    if not all(isinstance(i, int) and 0 <= i < count for i in faces):
        raise ValueError('Asset contains an invalid triangle vertex')
    triangles = len(faces) // 3
    if len(model['face_labels']) != triangles:
        raise ValueError('Triangle labels do not match triangle count')
    labels = {p['id'] for p in model['parts']}
    if not set(model['face_labels']).issubset(labels):
        raise ValueError('Triangle label has no corresponding part')
    for polygon in model.get('polygons', []):
        if len(polygon) < 3 or any(i < 0 or i >= count for i in polygon):
            raise ValueError('Asset contains an invalid polygon')
    return dict(vertices=count,triangles=triangles,polygons=len(model.get('polygons', [])),parts=len(labels))
