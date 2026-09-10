"""Inspect Blender source assets without modifying them."""
import bpy, json, sys
from mathutils import Vector
bpy.ops.wm.open_mainfile(filepath=sys.argv[-1], load_ui=False)
result = []
for obj in bpy.data.objects:
    if obj.type != 'MESH': continue
    vv = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not vv: continue
    result.append(dict(name=obj.name,vertices=len(vv),polygons=len(obj.data.polygons),lo=[min(v[i] for v in vv) for i in range(3)],hi=[max(v[i] for v in vv) for i in range(3)],materials=[m.name if m else None for m in obj.data.materials],modifiers=[dict(name=m.name,type=m.type) for m in obj.modifiers],hide=obj.hide_render))
print(json.dumps(dict(filepath=bpy.data.filepath,units=bpy.context.scene.unit_settings.system,scale=bpy.context.scene.unit_settings.scale_length,objects=result),indent=2))
