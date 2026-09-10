"""30 semantic design modules over a connected exterior body skin.

The source is our authored concept, not an OEM vehicle. Cosmetic material slots
are separate from semantic modules. Follower bindings have vertex identities;
any topology change must remap or invalidate them.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .model import Model, array_hash
from .detailed import _build
from .geometry import edges_and_counts


def _groups():
    groups=[
        ('hood','发动机盖',['hood','hood_trim']),
        ('roof','车顶外框',['roof','roof_trim']),
        ('front_bumper','前保险杠与进气饰面',['front_bumper','front_grille','lower_grille','vent_l','vent_r']),
        ('rear_bumper','后保险杠',['rear_bumper','rear_plate']),
        ('tailgate','尾门与后甲板',['tailgate','rear_trim']),
        ('windshield','前风挡',['windshield']),
        ('rear_glass','后风挡',['rear_glass']),
        ('panorama','全景车顶',['panorama']),
        ('splitter','前下唇',['splitter']),
        ('spoiler','尾部扰流唇',['spoiler']),
        ('diffuser','后扩散器区域',['diffuser']),
        ('underbody','底部护板',['underbody']),
    ]
    for suffix,word in [('r','右'),('l','左')]:
        for key,name,members in [
            ('front_fender','前翼子板',['front_fender','liner_f']),
            ('front_door','前车门',['front_door','handles','side_skirt']),
            ('rear_door','后车门',['rear_door']),
            ('rear_quarter','后翼子板',['rear_quarter','liner_b']),
            ('side_glass','侧窗与柱框',['front_window','rear_window','quarter_window','pillar']),
            ('mirror','后视镜总成',['mirror','mirror_glass']),
        ]:
            keys=[f'{m}{suffix}' if m.endswith(('_f','_b')) else f'{m}_{suffix}' for m in members]
            groups.append((f'{key}_{suffix}',word+name,keys))
    for axle,aname in [('f','前'),('b','后')]:
        for suffix,word in [('r','右'),('l','左')]:
            groups.append((f'wheel_{axle}{suffix}',word+aname+'轮组',
                           [f'{k}_{axle}{suffix}' for k in ['tire','rim','brake']]))
    groups += [('front_lights','前灯总成',['headlamp_l','headlamp_r','drl_l','drl_r']),
               ('rear_lights','尾灯总成',['taillamp_l','taillamp_r'])]
    return groups


def _build30():
    src=_build(design_skin=True)
    # Door shutlines remain a rendering concern, not tiny disconnected design solids.
    source_by_key={p['key']:p for p in src.parts}
    remove_keys={'seals','hood_trim','roof_trim','rear_trim'}
    keep=~np.isin(src.labels,[source_by_key[k]['id'] for k in remove_keys])
    oldf=src.faces[keep];oldl=src.labels[keep]
    used,inv=np.unique(oldf,return_inverse=True)
    v=src.vertices[used];f=inv.reshape(-1,3).astype(np.int32)
    parts=[];mapping={};colors={};palette=[];matids={}
    for p in src.parts:
        key=tuple(sorted(p['material'].items()))
        if key not in matids: matids[key]=len(palette);palette.append(p['material'])
        colors[p['id']]=matids[key]
    face_mats=[colors[int(i)] for i in oldl]
    for i,(key,name,members) in enumerate(_groups()):
        source=source_by_key[members[0]]
        p={k:val for k,val in source.items() if k not in ['pair_id','pair_key','follower_ids','followers','face_count']}
        p.update(id=i,key=key,name=name,source_members=members,locked=key.startswith('wheel_'))
        if key.startswith('wheel_'):p.update(category='wheels',category_name='轮组')
        parts.append(p)
        for member in members: mapping[source_by_key[member]['id']]=i
    assert set(np.unique(oldl)).issubset(mapping)
    labels=np.array([mapping[int(i)] for i in oldl],np.int32)
    bykey={p['key']:p['id'] for p in parts}
    for p in parts:
        if p['key'].endswith(('r','l')) and not p['key'].endswith(('diffuser','splitter')):
            k=p['key'][:-1]+('l' if p['key'][-1]=='r' else 'r')
            if k in bykey:p['pair_id']=bykey[k]
    # Connected topology, not label adjacency, defines the deformation skin.
    e,_=edges_and_counts(f)
    g=coo_matrix((np.ones(len(e)*2),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(len(v),len(v))).tocsr()
    n,cc=connected_components(g,directed=False)
    counts=np.bincount(cc);main=int(counts.argmax())
    skin_ids=np.flatnonzero(cc==main)
    followers=[]
    # Explicit, authored followers, fitted as a rigid component per disconnected island.
    # The other disconnected ornamental meshes stay fixed unless bound explicitly.
    follower_source_keys={'handles_l','handles_r','mirror_l','mirror_r','mirror_glass_l','mirror_glass_r',
                          'drl_l','drl_r','taillamp_l','taillamp_r','spoiler'}
    for component in range(n):
        ids=np.flatnonzero(cc==component)
        if component==main:continue
        mask=np.any(np.isin(f,ids),axis=1)
        keys={src.parts[int(i)]['key'] for i in np.unique(oldl[mask])}
        if not keys.intersection(follower_source_keys): continue
        parent_keys=None
        key=sorted(keys)[0]
        if 'handles' in key:
            side='l' if key.endswith('l') else 'r'
            parent_keys=[('front_door_' if v[ids,0].mean()<.5 else 'rear_door_')+side]
            labels[mask]=bykey[parent_keys[0]]
        elif 'mirror' in key:
            side='l' if key.endswith('l') else 'r';parent_keys=['front_door_'+side,'side_glass_'+side]
        elif 'drl' in key:parent_keys=['front_lights','hood','front_bumper']
        else:parent_keys=['tailgate','rear_quarter_l','rear_quarter_r']
        followers.append(dict(name=f'{key}:{component}',vertices=ids.tolist(),
                              parent_parts=[bykey[k] for k in parent_keys],
                              mode='rigid_kabsch',max_binding_distance_m=.35))
    # A mirror housing, stalk and glass are one rigid assembly, not independent fits.
    for side in ('l','r'):
        grouped=[b for b in followers if b['name'].startswith((f'mirror_{side}:',f'mirror_glass_{side}:'))]
        if grouped:
            keep_names={b['name'] for b in grouped}
            followers=[b for b in followers if b['name'] not in keep_names]
            item=dict(grouped[0]);item['name']=f'mirror_assembly_{side}'
            item['vertices']=sorted({i for b in grouped for i in b['vertices']})
            followers.append(item)
    meta=dict(src.metadata)
    meta.update(name='AeroGT · 30 区连续设计外壳',family_id='aerogt-design30-v03',
                construction='Authored shared body skin with wheel-house ceilings and closed diffuser boundary; regularized station spacing.',
                base_geometry_hash=array_hash(v,f),face_material_ids=face_mats,material_palette=palette,
                driver_vertices=skin_ids.tolist(),rigid_followers=followers,
                geometry_role='closed_edge_incidence_research_assembly_not_boolean_union',
                design_sets={
                    'roof':['roof','panorama','side_glass_l','side_glass_r'],
                    'rear':['tailgate','rear_glass','spoiler','rear_quarter_l','rear_quarter_r','rear_lights'],
                    'bonnet':['hood','front_fender_l','front_fender_r'],
                    'mirrors':['mirror_l','mirror_r'],
                    'underbody':['underbody','diffuser']},
                notes=['Original concept, not an OEM/DrivAer geometry.',
                       'Wheel cavities and diffuser are explicitly authored closures, not an automatic fill-all-holes algorithm.',
                       'Closed edge incidence does not imply a valid union, no self-intersection, correct gap, or CFD readiness.',
                       'Cosmetic material slots are independent of the 30 semantic design labels.'])
    return Model(v,f,labels,parts,meta)


@lru_cache(maxsize=1)
def _reference30():return _build30()


def design_model():return _reference30().clone()
