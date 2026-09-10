import bpy,json,sys
bpy.ops.wm.open_mainfile(filepath=sys.argv[-1],load_ui=False)
print('SCENE',[(c.name,c.hide_render,c.hide_viewport) for c in bpy.data.collections])
for o in bpy.data.objects:
 if o.type=='MESH':print(o.name,'scene',o.name in bpy.context.scene.objects,'collections',[c.name for c in o.users_collection],'visible',o.visible_get(),'hide',o.hide_get(),'display',o.display_type,'location',list(o.location),'scale',list(o.scale), 'modifiers',[(m.name,getattr(m,'levels',None),m.show_viewport,m.show_render) for m in o.modifiers])
for m in bpy.data.materials:
 print('MATERIAL',m.name, list(m.diffuse_color))
 if m.node_tree:
  for n in m.node_tree.nodes:
   if n.type in ('BSDF_PRINCIPLED','BSDF_DIFFUSE','BSDF_GLOSSY','TEX_IMAGE'):
    print('NODE',n.type,n.name,'IMAGE',n.image.filepath if n.type=='TEX_IMAGE' and n.image else '', 'INPUTS',[(i.name,list(i.default_value) if hasattr(i.default_value,'__len__') else i.default_value) for i in n.inputs if hasattr(i,'default_value') and i.name in ['Base Color','Color','Metallic','Roughness','Alpha']])
