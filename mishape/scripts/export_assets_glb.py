"""Export UV-preserving GLBs from the evaluated polygon archive, without bpy.

Requires the main project's numpy + trimesh environment. Studio PBR values are
used for viewing, while all source texture files stay alongside source.blend.
"""
from __future__ import annotations
import gzip,json
from pathlib import Path
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

ROOT=Path(__file__).resolve().parents[1]/'assets'

def main():
 catalog=json.loads((ROOT/'catalog.json').read_text())
 for entry in catalog['assets']:
  with gzip.open(ROOT/entry['full_mesh_path'],'rt') as f:model=json.load(f)
  v=np.array(model['vertices']).reshape(-1,3);faces=np.array(model['faces']).reshape(-1,3);poly_ids=np.array(model['source_face_ids']);mat_ids=np.array(model['face_materials']);labels=np.array(model['face_labels'])
  scene=trimesh.Scene();palette=model['metadata']['material_palette'];partmap={p['id']:p for p in model['parts']}
  for obj,uv_obj in zip(model['metadata']['source_objects'],model['uv_layers']):
   start=obj['polygon_start'];end=start+obj['polygon_count'];fi=np.flatnonzero((poly_ids>=start)&(poly_ids<end))
   if not len(fi):continue
   uv_array=np.array(next(iter(uv_obj['layers'].values()))).reshape(-1,2) if uv_obj['layers'] else None
   for mat_id in np.unique(mat_ids[fi]):
    for label in np.unique(labels[fi]):
     selected=fi[(mat_ids[fi]==mat_id)&(labels[fi]==label)]
     if not len(selected):continue
     coords=v[faces[selected]].reshape(-1,3);tt=np.arange(len(coords)).reshape(-1,3)
     material=palette[int(mat_id)];rgb=[int(material['base_color'][i:i+2],16) for i in (1,3,5)]+[255]
     pbr=PBRMaterial(name=material['name'],baseColorFactor=rgb,metallicFactor=material['metallic'],roughnessFactor=material['roughness'],doubleSided=True)
     uvs=[]
     if uv_array is not None:
      for face_index in selected:
       poly_index=int(poly_ids[face_index]);polygon=model['polygons'][poly_index];loops=uv_obj['polygon_loop_indices'][poly_index-start]
       for vi in faces[face_index]:uvs.append(uv_array[loops[polygon.index(int(vi))]])
     mesh=trimesh.Trimesh(vertices=coords,faces=tt,process=False,visual=TextureVisuals(uv=np.array(uvs) if uvs else None,material=pbr))
     key=partmap[int(label)]['key'];mesh.metadata.update(source_object=obj['name'],part=key,source_material=material['name'],uv_preserved=bool(uvs))
     scene.add_geometry(mesh,node_name=f'{key}/{obj["name"]}/{material["name"]}',geom_name=f'{obj["name"]}_{mat_id}_{label}')
  scene.metadata.update(asset_id=entry['id'],units='m',front_axis='-X',up_axis='+Z',scale_status='estimated',textures='Original texture files are retained alongside source.blend; GLB has studio PBR preview materials and source UVs.')
  path=ROOT/entry['id']/'model.glb';path.write_bytes(scene.export(file_type='glb'));entry['glb_path']=f'{entry["id"]}/model.glb';print(entry['id'],len(scene.geometry),'meshes',path.stat().st_size,'bytes')
 (ROOT/'catalog.json').write_text(json.dumps(catalog,separators=(',',':'),ensure_ascii=False))

if __name__=='__main__':main()
