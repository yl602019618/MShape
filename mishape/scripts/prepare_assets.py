"""Offline conversion of the two user-supplied Blender vehicles.

Run with a Python environment containing bpy, numpy and Pillow:
  python mishape/scripts/prepare_assets.py

Inputs are read-only. A verbatim .blend copy, all supplied textures, original
object polygons/UVs, an evaluated full mesh, and a compact viewport are retained.
Preview materials are explicit studio approximations: legacy Blender Internal
shaders do not round-trip to modern PBR. Source diffuse values are also recorded.
"""
from __future__ import annotations
import gzip, hashlib, json, math, shutil
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1] / 'assets'
SOURCE_ROOT = Path('/Users/zhijunzeng/Documents/Car')
ASSETS = [
 dict(id='porsche-930',name='Porsche 911 · 930 Turbo',subtitle='1975 · Imported polygon asset',source='free-1975-porsche-911-930-turbo',blend='911_scene.blend',length=4.30,paint='#aab9b3',ground='Plane.015',coat='Circle.000',wheel_objects=['Circle.037','Circle.038','Circle.048','Circle.050','Circle.051'],windshield='Circle.041',rear_glass='Circle.018',antenna=['Cylinder.002','Cylinder.006']),
 dict(id='porsche-carrera-4s',name='Porsche 911 · Carrera 4S',subtitle='Carrera 4S · Imported polygon asset',source='free-porsche-911-carrera-4s',blend='9e528d02fb594880b6f241f668d63bc0.blend',length=4.43,paint='#8c91ad',ground='Plane',coat='boot.011',wheel_objects=['Cylinder.000','Cylinder.001'],windshield='windshield',rear_glass='window_rear',antenna=[]),
]

def dump(path, data):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.suffix=='.gz':
  with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as f:json.dump(data,f,separators=(',',':'),ensure_ascii=False,allow_nan=False)
 else:path.write_text(json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False))

def rgb_hex(linear):
 def srgb(c):return 12.92*c if c<=.0031308 else 1.055*c**(1/2.4)-.055
 return '#'+''.join(f'{round(max(0,min(1,srgb(float(c))))*255):02x}' for c in linear[:3])

def palette_material(mat, spec):
 name=mat.name if mat else 'unassigned';low=name.lower()
 diffuse=list(mat.diffuse_color) if mat else [.5,.5,.5,1]
 color=rgb_hex(diffuse);rough=.48;metal=.08
 if low=='paint':color=spec['paint'];rough=.26;metal=.52
 elif any(s in low for s in ['rubber','tire']):color='#24292f';rough=.84;metal=0
 elif any(s in low for s in ['plastic','black']):color='#30373e';rough=.68;metal=.04
 elif any(s in low for s in ['silver','chrome','rim']):color='#aeb6c1';rough=.22;metal=.85
 elif low in ('window','glass'):
  color='#cadbe4' if low=='glass' and spec['id']=='porsche-carrera-4s' else '#263b44';rough=.13;metal=.25
 elif 'light' in low:color='#e4ebee';rough=.18;metal=.15
 elif low in ('logo','plate','license') or 'sticker' in low:color='#d4d8d9';rough=.35;metal=.05
 return dict(name=name,base_color=color,roughness=rough,metallic=metal,source_diffuse_color=diffuse,preview_approximation=True)

def extract_original(objects):
 result=[]
 for obj in objects:
  mesh=obj.data
  layers={uv.name:[round(c,7) for item in uv.data for c in item.uv] for uv in mesh.uv_layers}
  result.append(dict(name=obj.name,visible=obj.visible_get(),collections=[c.name for c in obj.users_collection],matrix_world=[list(row) for row in obj.matrix_world],vertices=[float(c) for v in mesh.vertices for c in v.co],polygons=[list(p.vertices) for p in mesh.polygons],polygon_materials=[p.material_index for p in mesh.polygons],polygon_loop_indices=[list(p.loop_indices) for p in mesh.polygons],materials=[m.name if m else None for m in mesh.materials],uv_layers=layers,modifiers=[dict(name=m.name,type=m.type,viewport=m.show_viewport,render=m.show_render,levels=getattr(m,'levels',None)) for m in obj.modifiers]))
 return result

def evaluate_mesh(obj, preview):
 # Keep topology connected; hard-edge normals are recalculated by the viewer.
 # Full evaluated data respects source subdivision. Preview caps subdivision at 1.
 saved=[]
 for mod in obj.modifiers:
  saved.append((mod,mod.show_viewport,getattr(mod,'levels',None)))
  if mod.type=='EDGE_SPLIT':mod.show_viewport=False
  if preview and mod.type=='SUBSURF':mod.levels=min(mod.levels,1)
 bpy.context.view_layer.update()
 evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
 mesh=bpy.data.meshes.new_from_object(evaluated,preserve_all_data_layers=True,depsgraph=bpy.context.evaluated_depsgraph_get())
 for mod,show,level in saved:
  mod.show_viewport=show
  if level is not None:mod.levels=level
 return mesh

def semantic(objname,matname,center,spec):
 low=matname.lower();x,y,z=center
 if objname in spec['wheel_objects']:
  return ('wheel_front_' if x<0 else 'wheel_rear_')+('left' if y>0 else 'right')
 if objname==spec['windshield']:return 'windshield'
 if objname==spec['rear_glass']:return 'rear_glass'
 if 'glass' in low or low=='window':return 'glazing'
 if low=='paint':
  if objname=='Circle.002':return 'hood'
  return 'body_shell'
 if 'light' in low or low=='tex_shiny':return 'front_lights' if x<0 else 'rear_lights'
 if any(k in low for k in ['chrome','silver']):return 'brightwork'
 if 'sticker' in low or low in ('plate','license','logo'):return 'badges_and_plates'
 if objname in spec['antenna']:return 'antenna'
 if any(k in low for k in ['plastic','black']):return 'trim_and_underbody'
 return 'details'

PART_NAMES={'body_shell':'车身外壳 / Body shell','hood':'前舱盖 / Hood','windshield':'前风挡 / Windshield','rear_glass':'后风挡 / Rear glass','glazing':'侧窗 / Glazing','brightwork':'金属饰件 / Brightwork','front_lights':'前灯 / Front lights','rear_lights':'后灯 / Rear lights','badges_and_plates':'徽标与牌照 / Badges','trim_and_underbody':'饰件与底盘 / Trim','antenna':'天线 / Antenna','details':'细节 / Details'}
for axle in ['front','rear']:
 for side in ['left','right']:
  key=f'wheel_{axle}_{side}';PART_NAMES[key]=f'{"前" if axle=="front" else "后"}{"左" if side=="left" else "右"}车轮 / {key}'

def build_model(objects,spec,transform,palette,preview):
 vertices=[];faces=[];labels=[];materials=[];polygons=[];polygon_parts=[];polygon_materials=[];source_face_ids=[];parts=[];part_ids={};source_objects=[];uv_layers=[];export_objects=[]
 for obj in objects:
  mesh=evaluate_mesh(obj,preview)
  # Retain a moderate polygon budget per object with Blender's quad dissolve or
  # collapse only for very dense source parts. Source/full polygon data is kept.
  tri_est=sum(len(p.vertices)-2 for p in mesh.polygons)
  paint_object=any(m and m.name=='paint' for m in mesh.materials)
  budget=40000 if paint_object else 5000
  if preview and tri_est>budget:
   temp=bpy.data.objects.new('_MiShape_decimate',mesh);bpy.context.collection.objects.link(temp)
   modifier=temp.modifiers.new('MiShape viewport LOD','DECIMATE');modifier.ratio=max(.08,min(1,budget/tri_est));modifier.use_collapse_triangulate=False
   bpy.context.view_layer.update()
   reduced=bpy.data.meshes.new_from_object(temp.evaluated_get(bpy.context.evaluated_depsgraph_get()),preserve_all_data_layers=True,depsgraph=bpy.context.evaluated_depsgraph_get())
   bpy.data.objects.remove(temp,do_unlink=True);bpy.data.meshes.remove(mesh);mesh=reduced
  offset=len(vertices)//3;poly_offset=len(polygons)
  vv=np.array([list(transform @ obj.matrix_world @ v.co) for v in mesh.vertices],dtype=float)
  vertices.extend(np.round(vv,7).reshape(-1).tolist())
  mats=[m.name if m else 'unassigned' for m in mesh.materials]
  poly_labels=[]
  for p in mesh.polygons:
   mi=min(p.material_index,max(0,len(mats)-1));mn=mats[mi] if mats else 'unassigned'
   center=vv[list(p.vertices)].mean(axis=0)
   if 'light' in mn.lower() and center[0]>1.0:
    name=mn+'_rear_preview'
    if name not in palette:palette[name]={**palette[mn],'name':name,'base_color':'#a83c38','source_material':mn}
    mn=name
   key=semantic(obj.name,mn,center,spec)
   if key not in part_ids:
    pid=len(parts);part_ids[key]=pid;pm=palette[mn]
    parts.append(dict(id=pid,key=key,name=PART_NAMES[key],color=pm['base_color'],material=pm,rigid=key.startswith('wheel_'),locked=False,source='source-object/material grouping'))
   pid=part_ids[key];poly_labels.append(pid)
   polygons.append([int(i)+offset for i in p.vertices]);polygon_parts.append(pid);polygon_materials.append(mn)
  mesh.calc_loop_triangles()
  for t in mesh.loop_triangles:
   faces.extend([int(i)+offset for i in t.vertices]);labels.append(poly_labels[t.polygon_index]);materials.append(polygon_materials[poly_offset+t.polygon_index]);source_face_ids.append(poly_offset+t.polygon_index)
  source_objects.append(dict(name=obj.name,vertex_start=offset,vertex_count=len(vv),polygon_start=poly_offset,polygon_count=len(mesh.polygons),source_materials=mats))
  if not preview:
   uv_layers.append(dict(source_object=obj.name,layers={uv.name:[round(c,7) for item in uv.data for c in item.uv] for uv in mesh.uv_layers},polygon_loop_indices=[list(p.loop_indices) for p in mesh.polygons]))
  bpy.data.meshes.remove(mesh)
 palette_names=list(palette);palette_ids={name:i for i,name in enumerate(palette_names)}
 v=np.array(vertices).reshape(-1,3)
 # Evaluate modifiers before calibration; remove any residual LOD ground offset.
 v[:,2]-=v[:,2].min();vertices=np.round(v,7).reshape(-1).tolist();bounds=[v.min(0).tolist(),v.max(0).tolist()]
 wheel_centers=[]
 for part in parts:
  if not part['key'].startswith('wheel_'):continue
  ids=np.unique(np.array(faces).reshape(-1,3)[np.array(labels)==part['id']]);points=v[ids];center=(points.min(0)+points.max(0))/2
  part['center']=center.tolist();part['wheel_radius_m']=float((points[:,2].max()-points[:,2].min())/2);wheel_centers.append((part['key'],center))
 landmarks={}
 for axle in ('front','rear'):
  points=[c for name,c in wheel_centers if f'wheel_{axle}_' in name]
  if points:landmarks[f'{axle}_axle']=np.mean(points,axis=0).tolist();landmarks[f'{axle}_axle'][1]=0
 for name,objectname in [('windshield',spec['windshield']),('rear_glass',spec['rear_glass'])]:
  rec=next((r for r in source_objects if r['name']==objectname),None)
  if rec:
   points=v[rec['vertex_start']:rec['vertex_start']+rec['vertex_count']];zlo,zhi=np.quantile(points[:,2],[.08,.92]);bottom=points[points[:,2]<=zlo].mean(0);top=points[points[:,2]>=zhi].mean(0);bottom[1]=0;top[1]=0;landmarks[f'{name}_base']=bottom.tolist();landmarks[f'{name}_top']=top.tolist()
 metadata=dict(asset_id=spec['id'],source='user-provided Blender asset',units='m',front_axis='-X',up_axis='+Z',coordinate_system='right-handed; X longitudinal, Y width, Z up',scale_status='estimated',scale_confidence='requires known dimension calibration',reference_length_m=spec['length'],bounds=bounds,dimensions_m=(v.max(0)-v.min(0)).tolist(),material_palette=[palette[n] for n in palette_names],landmarks=landmarks,landmark_provenance='wheel and glazing source-object bounds; approximate',source_objects=source_objects,polygon_preservation='Original source control polygons in original_mesh.json.gz; full evaluated polygons in fullmesh.json.gz; display LOD polygons here',display_lod=preview,preview_materials='Studio PBR approximations; original diffuse values and supplied textures preserved',original_mesh_path=f'{spec["id"]}/original_mesh.json.gz',full_mesh_path=f'{spec["id"]}/fullmesh.json.gz')
 model=dict(name=spec['name'],vertices=vertices,faces=faces,face_labels=labels,face_materials=[palette_ids[m] for m in materials],source_face_ids=source_face_ids,parts=parts,polygons=polygons,polygon_labels=polygon_parts,polygon_materials=[palette_ids[m] for m in polygon_materials],metadata=metadata)
 if not preview:model['uv_layers']=uv_layers
 return model

def main():
 entries=[]
 for spec in ASSETS:
  source=SOURCE_ROOT/spec['source'];blend=source/'source'/spec['blend'];dest=ROOT/spec['id'];dest.mkdir(parents=True,exist_ok=True)
  bpy.ops.wm.open_mainfile(filepath=str(blend),load_ui=False)
  objects=[o for o in bpy.data.objects if o.type=='MESH'];original=extract_original(objects)
  included=[];excluded=[]
  for o in objects:
   reason=None
   if not o.visible_get():reason='hidden source collection/reference object'
   elif o.name==spec['ground']:reason='baked shadow ground plane'
   elif o.name==spec['coat']:reason='duplicate transparent clearcoat shell; represented as paint response'
   elif o.name in spec['antenna']:reason='decorative antenna omitted from design envelope; preserved in original mesh'
   if reason:excluded.append(dict(name=o.name,reason=reason))
   else:included.append(o)
  evaluated_points=[]
  for o in included:
   mesh=evaluate_mesh(o,False);evaluated_points.extend([list(o.matrix_world @ v.co) for v in mesh.vertices]);bpy.data.meshes.remove(mesh)
  points=np.array(evaluated_points);lo=points.min(0);hi=points.max(0)
  scale=spec['length']/(hi[1]-lo[1]);center=(lo+hi)/2
  transform=Matrix(((0,scale,0,-center[1]*scale),(-scale,0,0,center[0]*scale),(0,0,scale,-lo[2]*scale),(0,0,0,1)))
  palette={m.name:palette_material(m,spec) for m in bpy.data.materials};palette['unassigned']=palette_material(None,spec)
  texture_files=[]
  (dest/'textures').mkdir(exist_ok=True)
  for p in sorted((source/'textures').iterdir()):
   if p.is_file():shutil.copy2(p,dest/'textures'/p.name);texture_files.append(dict(name=p.name,path=f'textures/{p.name}',bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  shutil.copy2(blend,dest/'source.blend')
  dump(dest/'original_mesh.json.gz',dict(schema='mishape.source-polygons.v1',source_blend_name=blend.name,source_sha256=hashlib.sha256(blend.read_bytes()).hexdigest(),source_units=bpy.context.scene.unit_settings.system,source_unit_scale=bpy.context.scene.unit_settings.scale_length,normalization_matrix=[list(row) for row in transform],objects=original,materials=list(palette.values()),textures=texture_files,excluded_from_design=excluded))
  print(spec['id'],'building full',flush=True)
  full=build_model(included,spec,transform,palette,False);full['metadata']['excluded_source_objects']=excluded;full['metadata']['textures']=texture_files
  dump(dest/'fullmesh.json.gz',full)
  print(spec['id'],'building viewport',flush=True)
  preview=build_model(included,spec,transform,palette,True);preview['metadata']['excluded_source_objects']=excluded;preview['metadata']['textures']=texture_files
  dump(dest/'viewport.json',preview)
  entry=dict(id=spec['id'],name=spec['name'],subtitle=spec['subtitle'],path=f'{spec["id"]}/viewport.json',full_mesh_path=f'{spec["id"]}/fullmesh.json.gz',original_mesh_path=f'{spec["id"]}/original_mesh.json.gz',source_blend_path=f'{spec["id"]}/source.blend',dimensions_m=preview['metadata']['dimensions_m'],vertices=len(preview['vertices'])//3,triangles=len(preview['faces'])//3,polygons=len(preview['polygons']),parts=len(preview['parts']),source_polygon_count=sum(len(o['polygons']) for o in original),scale_status='estimated',reference_length_m=spec['length'],front_axis='-X',color=spec['paint'],provenance='User supplied local asset; no redistribution license inferred',excluded=excluded)
  entries.append(entry);dump(ROOT/'catalog.json',dict(schema='mishape.assets.v1',assets=entries));print(json.dumps(entry),flush=True)

if __name__=='__main__':main()
