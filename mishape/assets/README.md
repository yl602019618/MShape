# MiShape local polygon assets

These are local conversions of the two user-provided Blender vehicles. Source folders are untouched. This directory is not a claim of ownership or a license to redistribute those vehicles.

Each asset contains:

- `viewport.json`: connected polygon display LOD, triangulation, source polygon indices, material indices, semantic parts and approximate landmarks. The 3D viewer can consume this directly.
- `fullmesh.json.gz`: evaluated geometry with source subdivisions, polygon topology, per-loop UVs and source object ranges. Edge-split modifiers are represented through shading rather than disconnected topology.
- `original_mesh.json.gz`: all original source objects, including excluded scene props, their unmodified local coordinates, polygon topology, per-loop UVs, world matrices and modifier metadata.
- `source.blend`: a byte-for-byte copy of the input. Its SHA-256 is in `original_mesh.json.gz`.
- `textures/`: all supplied texture images without modification.
- `model.glb`: evaluated geometry, component/material groups, source UVs and approximate PBR materials. It retains MiShape's Z-up coordinates; consumers expecting glTF Y-up should rotate -90 degrees about X.

The coordinate transform is X = original Y, Y = negative original X, Z = original Z. The front is negative X; the floor is Z = 0; coordinates are in meters. Known dimensions were not supplied. The 930 is normalized to approximately 4.30 m long and the Carrera 4S to approximately 4.43 m; both are explicitly marked `scale_status: estimated`. The small viewport LOD can differ from the full mesh bounding box by fractions of a millimeter. Width/height are measurements after this uniform calibration, not verified manufacturer dimensions.

Excluded from the design envelope: shadow planes, hidden reference geometry, duplicate transparent clearcoat shells, and the 930's decorative antenna. All are preserved in the original archive and Blend file. Clearcoat is approximated through the paint's material response. Legacy shader graphs do not directly map to PBR: source diffuse values are recorded, and the viewport/GLB uses studio paint, glass, light and metal approximations. Original image maps and UVs are preserved for a future texture renderer; the current viewer does not render those images.

Wheel groups are explicitly labelled `wheel_front_left`, `wheel_front_right`, `wheel_rear_left`, `wheel_rear_right` and marked rigid. Their positions are obtained from source-object bounds. Windshield and rear-glass landmarks also come from source objects and are approximate; they can be calibrated by the application.

Offline regeneration (requires Python with bpy 5.2, numpy):

```sh
python mishape/scripts/prepare_assets.py
# The main project environment supplies trimesh for GLB export:
.venv/bin/python mishape/scripts/export_assets_glb.py
.venv/bin/python -m pytest mishape/tests/test_assets.py -q
```

The production runtime does not depend on bpy or Blender. `mishape.assets_pipeline` uses the Python standard library to load catalog assets. The converter uses the original local source paths explicitly listed in `prepare_assets.py`.
