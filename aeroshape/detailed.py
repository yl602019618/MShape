"""Original, explicitly fictional exterior design reference.

This is a structured multi-panel surface, not an OEM/DrivAer geometry. All
coordinates are metres, X points rearward, Y leftward, Z upward. Its details
are real triangles (including wheel openings), not a textured picture.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.interpolate import PchipInterpolator
import trimesh
from .model import Model

MATERIALS = {
    'paint': dict(base_color='#49778e', roughness=.29, metallic=.72),
    'glass': dict(base_color='#162b38', roughness=.12, metallic=.35),
    'black': dict(base_color='#18212a', roughness=.57, metallic=.05),
    'rubber': dict(base_color='#171b21', roughness=.93, metallic=0.),
    'chrome': dict(base_color='#b2bdc6', roughness=.22, metallic=.92),
    'brake': dict(base_color='#58616b', roughness=.66, metallic=.65),
    'lamp': dict(base_color='#233343', roughness=.16, metallic=.45),
    'led': dict(base_color='#dbf3ff', roughness=.2, metallic=.1, emission=.65),
    'red': dict(base_color='#a62034', roughness=.2, metallic=.2, emission=.2),
}
GROUP_NAMES={'body':'车身外板', 'glazing':'玻璃与窗框', 'lighting':'灯具',
             'aero':'气动附件', 'wheels':'轮组', 'detail':'外饰细节'}

class Builder:
    def __init__(self):
        self.parts=[]; self.ids={}; self.vertices=[]; self.faces=[]; self.labels=[]; self.offset=0
    def part(self,key,name,group='body',mat='paint',locked=False,side=None,pair=None,followers=None):
        i=len(self.parts); self.ids[key]=i
        pal=['#89c5cb','#9fbbd1','#95c9b0','#d0b792','#b6b5d7','#8fabbc']
        p=dict(id=i,key=key,name=name,category=group,category_name=GROUP_NAMES[group],
               color=pal[i%len(pal)],material=dict(MATERIALS[mat]),locked=locked,side=side,
               pair_key=pair,followers=followers or [],pin_open_boundary=False)
        self.parts.append(p);return i
    def add(self,v,f,key):
        v=np.asarray(v,float);f=np.asarray(f,np.int32)
        if not len(f):return
        self.vertices.append(v);self.faces.append(f+self.offset)
        if isinstance(key,str): labs=np.full(len(f),self.ids[key],np.int32)
        else: labs=np.asarray(key,np.int32)
        self.labels.append(labs);self.offset+=len(v)
    def patch(self,uv, key, flip=False):
        v=np.asarray(uv);ni,nj=v.shape[:2]
        ii,jj=np.meshgrid(np.arange(ni-1),np.arange(nj-1),indexing='ij')
        a=(ii*nj+jj).ravel();b=a+1;c=a+nj;d=c+1
        f=np.stack((np.c_[a,c,b],np.c_[b,c,d]),axis=1).reshape(-1,3)
        if flip:f=f[:,[0,2,1]]
        if not isinstance(key,str): key=np.repeat(np.asarray(key).ravel(),2)
        self.add(v.reshape(-1,3),f,key)
    def mesh(self,m,key): self.add(m.vertices,m.faces,key)
    def finish(self,meta):
        v=np.vstack(self.vertices);f=np.vstack(self.faces);l=np.concatenate(self.labels)
        # Weld coincident seam vertices, NOT physical panel gaps. Do not join near surfaces.
        _,idx,inv=np.unique(np.round(v,9),axis=0,return_index=True,return_inverse=True)
        v=v[idx];f=inv[f]
        t=v[f];area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
        good=area>1e-11;f=f[good];l=l[good]
        used,inv=np.unique(f,return_inverse=True);v=v[used];f=inv.reshape(-1,3)
        for p in self.parts:
            p['face_count']=int((l==p['id']).sum())
            if p['pair_key']: p['pair_id']=self.ids[p['pair_key']]
            p['follower_ids']=[self.ids[k] for k in p['followers'] if k in self.ids]
        meta['construction_removed_degenerate_faces']=int((~good).sum())
        return Model(v,f,l,self.parts,meta)


def tube(path,radius=.003,sections=8):
    """Closed tube along a sampled polyline; rounds are geometry, not screen-space lines."""
    p=np.asarray(path,float);d=np.gradient(p,axis=0);d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-20)
    ref=np.tile([0.,0.,1.],(len(p),1));near=np.abs(d[:,2])>.92;ref[near]=[0,1,0]
    a=np.cross(d,ref);a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-20);b=np.cross(d,a)
    angle=np.arange(sections)*2*np.pi/sections
    v=p[:,None,:]+radius*(np.cos(angle)[None,:,None]*a[:,None,:]+np.sin(angle)[None,:,None]*b[:,None,:])
    ff=[]
    for i in range(len(p)-1):
        for j in range(sections):
            x=i*sections+j;y=i*sections+(j+1)%sections;z=(i+1)*sections+j;w=(i+1)*sections+(j+1)%sections
            ff.extend([[x,y,z],[y,w,z]])
    for end,base in ((0,0),(-1,(len(p)-1)*sections)):
        for j in range(1,sections-1):ff.append([base,base+j,base+j+1] if end==0 else [base,base+j+1,base+j])
    m=trimesh.Trimesh(v.reshape(-1,3),ff,process=False);m.fix_normals();return m


def radial_surface(x,y,z,profile,side=1,segments=96):
    """Surface of revolution around a wheel's lateral axis; profile=(axial,radius)."""
    p=np.asarray(profile,float);ang=np.arange(segments)*2*np.pi/segments
    v=np.empty((len(p),segments,3));v[:,:,0]=x+p[:,1,None]*np.cos(ang)
    v[:,:,1]=y+side*p[:,0,None];v[:,:,2]=z+p[:,1,None]*np.sin(ang)
    ff=[]
    for i in range(len(p)-1):
        for j in range(segments):
            a=i*segments+j;b=i*segments+(j+1)%segments;c=(i+1)*segments+j;d=(i+1)*segments+(j+1)%segments
            ff.extend([[a,b,c],[b,d,c]])
    m=trimesh.Trimesh(v.reshape(-1,3),ff,process=False);m.fix_normals();return m


def _build(design_skin=False) -> Model:
    b=Builder()
    # Explicit exterior semantics, independently selectable from material identity.
    for args in [
        ('hood','发动机盖','body','paint'),('roof','车顶外框','body','paint'),
        ('front_bumper','前保险杠','body','paint'),('rear_bumper','后保险杠','body','paint'),
        ('tailgate','尾门 / 后甲板','body','paint'),('windshield','前风挡','glazing','glass'),
        ('rear_glass','后风挡','glazing','glass'),('panorama','全景车顶玻璃','glazing','glass'),
        ('front_grille','前主格栅','aero','black'),('lower_grille','下进气口','aero','black'),
        ('splitter','前下唇','aero','black'),('spoiler','尾部扰流唇','aero','paint'),
        ('diffuser','后扩散器 / 导流鳍','aero','black'),('underbody','简化底部护板','aero','black'),
        ('seals','外板分缝 / 密封线','detail','black'),
        ('rear_plate','后牌照板 / 安装位','detail','chrome'),
        ('hood_trim','前盖分缝','detail','black'),
        ('roof_trim','车顶前后密封线','detail','black'),
        ('rear_trim','后玻璃下缘密封线','detail','black')]:b.part(*args)
    for s,word in [(-1,'右'),(1,'左')]:
        suf='r' if s<0 else 'l';other='l' if s<0 else 'r'
        specs=[('front_fender','前翼子板','body','paint'),('front_door','前车门外板','body','paint'),
               ('rear_door','后车门外板','body','paint'),('rear_quarter','后翼子板','body','paint'),
               ('side_skirt','侧裙','aero','black'),('front_window','前侧窗','glazing','glass'),
               ('rear_window','后侧窗','glazing','glass'),('quarter_window','后固定窗','glazing','glass'),
               ('pillar','A/B/C 柱与窗框','glazing','black'),('mirror','后视镜壳及支座','aero','paint'),
               ('mirror_glass','后视镜镜片','detail','chrome'),('headlamp','前灯壳','lighting','lamp'),
               ('drl','前日行灯','lighting','led'),('taillamp','尾灯','lighting','red'),
               ('handles','前后门把手','detail','chrome'),('vent','前侧导流口','aero','black')]
        for k,n,g,m in specs:
            followers=[f'handles_{suf}'] if k in ('front_door','rear_door') else []
            b.part(f'{k}_{suf}',f'{word}{n}',g,m,side=s,pair=f'{k}_{other}',followers=followers)
        for axle,axname in [('f','前'),('b','后')]:
            for k,n,mat in [('tire','轮胎','rubber'),('rim','轮毂及辐条','chrome'),('brake','制动盘','brake'),('liner','轮拱内衬','black')]:
                b.part(f'{k}_{axle}{suf}',f'{word}{axname}{n}','wheels' if k!='liner' else 'body',mat,
                       locked=k in ('tire','rim','brake'),side=s,pair=f'{k}_{axle}{other}')

    # Sculpted belt line, roof, bonnet and rear-screen profiles.
    knots=np.array([-2.45,-2.22,-1.7,-1.05,-.85,-.40,0,.68,.96,1.33,1.70,2.12,2.43])
    top=PchipInterpolator(knots,[.765,.855,.995,1.055,1.17,1.463,1.505,1.493,1.442,1.265,1.05,1.015,.965])
    belt=PchipInterpolator(knots,[.715,.825,.943,1.005,1.010,1.020,1.030,1.030,1.025,1.002,.975,.928,.889])
    width=PchipInterpolator(knots,[.80,.912,.978,.954,.942,.940,.935,.947,.971,.982,.963,.932,.862])
    profile=PchipInterpolator([0,.30,.59,.70,.77,.84,.94,1.],[1.,.999,.989,.969,.90,.68,.24,0.])
    def xe(x,r):
        x=np.asarray(x);r=np.asarray(r)
        return x+.155*r*r*np.clip((-2.02-x)/.43,0,1)**2-.10*r*r*np.clip((x-2.02)/.41,0,1)**2
    def skin(x,r):
        x,r=np.broadcast_arrays(x,r);ar=np.abs(r)
        z=belt(x)+(top(x)-belt(x))*profile(ar)
        # Bonnet character lines and subtle roof crown.
        z+=.014*np.exp(-((x+1.65)/.60)**4)*np.exp(-((ar-.55)/.06)**2)
        return np.stack((xe(x,r),width(x)*r,z),axis=-1)
    special=[-2.45,-2.26,-2.08,-1.06,-.89,-.40,-.25,.22,.30,.78,.94,1.32,1.62,1.74,2.13,2.43]
    xs=np.unique(np.r_[np.linspace(-2.45,2.43,145),special])
    for wx in [-1.49,1.43]:
        xs=np.unique(np.r_[xs,wx+np.linspace(-.405,.405,49)])
    rs=np.unique(np.r_[np.linspace(-1,1,97),-.985,-.80,-.77,-.64,.64,.77,.80,.985])
    if design_skin:
        # Prefer semantic breakpoints; never place two sampling stations almost coincident.
        def spaced(values, anchors, gap):
            result=list(sorted(set(anchors)))
            for value in sorted(values):
                if min(abs(value-a) for a in result)>=gap: result.append(float(value))
            return np.array(sorted(result))
        xs=spaced(xs,special+[-1.895,-1.085,1.025,1.835],.013)
        positive=spaced(np.abs(rs),[0.,.64,.77,.80,.985,1.],.010)
        rs=np.unique(np.r_[-positive,positive])
    X,R=np.meshgrid(xs,rs,indexing='ij');pts=skin(X,R)
    xc=(xs[:-1]+xs[1:])[:,None]/2;rc=(rs[:-1]+rs[1:])[None,:]/2
    xx,rr=np.broadcast_arrays(xc,rc);ar=np.abs(rr)
    labs=np.full(xx.shape,b.ids['hood'],np.int32)
    labs[xx>=-.40]=b.ids['roof'];labs[xx>=.94]=b.ids['tailgate']
    labs[(xx>=-1.06)&(xx<-.40)&(ar<.77)]=b.ids['windshield']
    labs[(xx>=.94)&(xx<1.62)&(ar<.77)]=b.ids['rear_glass']
    labs[(xx>=-.25)&(xx<.78)&(ar<.64)]=b.ids['panorama']
    for sign,suf in [(-1,'r'),(1,'l')]:
        sel=rr*sign>0
        labs[sel&(ar>.77)&(xx<-1.06)]=b.ids['front_fender_'+suf]
        pillar=sel&(ar>=.77)&(xx>=-.89)&(xx<1.62)
        labs[pillar]=b.ids['pillar_'+suf]
        w=pillar&(ar>.80)&(ar<.985)
        labs[w&(xx<.22)]=b.ids['front_window_'+suf]
        labs[w&(xx>=.30)&(xx<.94)]=b.ids['rear_window_'+suf]
        labs[w&(xx>=.94)&(xx<1.32)]=b.ids['quarter_window_'+suf]
        # Preserve a painted shoulder rail below the side windows.
        labs[sel&(ar>=.985)&(xx>=-.89)&(xx<1.62)]=b.ids['rear_quarter_'+suf]
        labs[sel&(xx<-2.08)&(xx>=-2.26)&(ar>.64)&(ar<.985)]=b.ids['headlamp_'+suf]
    b.patch(pts,labs)

    def lower(x):
        x=np.asarray(x);z=np.full_like(x,.225,dtype=float)
        for wx in [-1.49,1.43]:
            inside=np.abs(x-wx)<=.405
            z=np.maximum(z,np.where(inside,.355+np.sqrt(np.maximum(.405**2-(x-wx)**2,0)),.225))
        return z
    def side_skin(x,t,s):
        x,t=np.broadcast_arrays(x,t);z=lower(x)+(belt(x)-lower(x))*t
        # Longitudinal shoulder, concave door character and lower rocker flare.
        y=width(x)-.033*(1-t)+(.016*np.exp(-((t-.82)/.10)**2)-.020*np.exp(-((t-.37)/.18)**2))*np.sin(np.pi*t)
        return np.stack((xe(x,np.ones_like(x)),s*y,z),axis=-1)
    ts=np.linspace(0,1,23)
    for s,suf in [(-1,'r'),(1,'l')]:
        X,T=np.meshgrid(xs,ts,indexing='ij');p=side_skin(X,T,s)
        xmid=(X[:-1,:-1]+X[1:,1:])/2;zmid=(p[:-1,:-1,2]+p[1:,1:,2])/2
        labs=np.full(xmid.shape,b.ids['rear_quarter_'+suf],np.int32)
        labs[xmid<-1.06]=b.ids['front_fender_'+suf]
        labs[(xmid>=-1.06)&(xmid<.26)]=b.ids['front_door_'+suf]
        labs[(xmid>=.26)&(xmid<1.32)]=b.ids['rear_door_'+suf]
        labs[zmid<.32]=b.ids['side_skirt_'+suf]
        b.patch(p,labs,flip=s>0)
        # Actual wheel-well lips and inset liners; the exterior skin has openings.
        for wx,ax in [(-1.49,'f'),(1.43,'b')]:
            angle=np.linspace(0,np.pi,65)
            xx=wx+.405*np.cos(angle);zz=.355+.405*np.sin(angle)
            yy=width(xx)-.016
            path=np.c_[xx,s*yy,zz]
            b.mesh(tube(path,.009,8),'front_fender_'+suf if ax=='f' else 'rear_quarter_'+suf)
            q=np.empty((len(angle),7,3));q[:,:,0]=xx[:,None];q[:,:,2]=zz[:,None]
            q[:,:,1]=s*(yy[:,None]-np.linspace(0,.23,7)[None,:])
            if not design_skin: b.patch(q,f'liner_{ax}{suf}',flip=s<0)
        # Door gaps, belt-line and glass trims are thin physical curves.
        for bx in [-1.06,.26,1.32]:
            line=side_skin(np.full(24,bx),np.linspace(.05,1,24),s);line[:,1]+=s*.002
            b.mesh(tube(line,.0025,6),'seals')
        xline=np.linspace(-.88,1.57,110)
        for r in [.799,.987]:
            line=skin(xline,np.full(len(xline),s*r));line[:,2]+=.0025
            b.mesh(tube(line,.004,6),'pillar_'+suf)
        # Flush handles (one semantic module per side, not one per bolt).
        for cx in [.045,1.06]:
            p=side_skin(cx,.89,s)
            m=trimesh.creation.icosphere(subdivisions=2)
            m.vertices*=np.array([.076,.010,.013]);m.apply_translation(p+[0,s*.006,0]);b.mesh(m,'handles_'+suf)
        # Teardrop mirror with a geometrical stalk and rear-facing glass inset.
        m=trimesh.creation.icosphere(subdivisions=3);m.vertices*=np.array([.165,.100,.062])
        m.vertices[:,1]+= .035*(m.vertices[:,0]/.165)
        m.apply_translation([-.77,s*1.056,1.137]);
        if s<0: # Reflect a canonical left mirror rather than a handed distortion.
            m.vertices[:,1]= -m.vertices[:,1] - 2*1.056
        b.mesh(m,'mirror_'+suf)
        path=np.array([[-.86,s*.932,1.065],[-.845,s*.97,1.11],[-.79,s*1.015,1.13]])
        b.mesh(tube(path,.020,10),'mirror_'+suf)
        m=trimesh.creation.icosphere(subdivisions=2);m.vertices*=np.array([.009,.081,.047])
        m.apply_translation([-.622,s*1.064,1.139]);b.mesh(m,'mirror_glass_'+suf)
        # Daylight signature is placed exactly on the selected lamp surface.
        for lx in [-2.227,-2.115]:
            rr=np.linspace(.666,.965,30)*s;line=skin(np.full(30,lx),rr);line[:,2]+=.006
            b.mesh(tube(line,.006,8),'drl_'+suf)
        # Lateral air-curtain inset.
        yy=np.linspace(.66,.88,18)*s;vv=np.linspace(.28,.40,7)
        Y,Z=np.meshgrid(yy,vv,indexing='ij');X=-2.448+.16*(Y/.81)**2
        if not design_skin: b.patch(np.stack((X,Y,Z),-1),'vent_'+suf,flip=s>0)

    def end_surface(rear=False):
        # Share the exact boundary samples and edge profile with top/side skin.
        raw=2.43 if rear else -2.45;rr=rs;vv=ts
        R,V=np.meshgrid(rr,vv,indexing='ij');upper=skin(np.full_like(R,raw),R)
        P=np.empty(R.shape+(3,));P[:,:,0]=upper[:,:,0]+(.014 if rear else -.024)*np.sin(np.pi*V)*(1-R**2)
        P[:,:,1]=R*(width(raw)-.033*(1-V)+(.016*np.exp(-((V-.82)/.10)**2)-.020*np.exp(-((V-.37)/.18)**2))*np.sin(np.pi*V))
        lower_z=.225
        if design_skin and rear:
            t=np.clip((np.abs(R)-.65)/.20,0,1)
            lower_z=.225+.10*(1-t*t*(3-2*t))
        P[:,:,2]=lower_z+V*(upper[:,:,2]-lower_z)
        mid=(P[:-1,:-1]+P[1:,1:])/2
        labs=np.full(mid.shape[:2],b.ids['rear_bumper' if rear else 'front_bumper'],np.int32)
        if not rear:
            labs[(np.abs(mid[:,:,1])<.58)&(mid[:,:,2]>.49)&(mid[:,:,2]<.66)]=b.ids['front_grille']
            labs[(np.abs(mid[:,:,1])<.60)&(mid[:,:,2]>.29)&(mid[:,:,2]<.43)]=b.ids['lower_grille']
        else:
            labs[mid[:,:,2]>.74]=b.ids['tailgate']
            labs[mid[:,:,2]<.46]=b.ids['diffuser']
        b.patch(P,labs,flip=not rear)
    end_surface(False);end_surface(True)
    # Neutral plate recess, without logos or a claimed production registration.
    m=trimesh.creation.box(extents=[.013,.595,.165]);m.apply_translation([2.452,0,.639]);b.mesh(m,'seals')
    m=trimesh.creation.box(extents=[.003,.525,.111]);m.apply_translation([2.460,0,.639]);b.mesh(m,'rear_plate')
    # Grille slats. A shallow insert, not a certified flow-through engine bay.
    for z in [.326,.363,.400,.535,.574,.613]:
        yy=np.linspace(-.565,.565,40);xx=-2.45+.155*(yy/.80)**2-.025
        b.mesh(tube(np.c_[xx,yy,np.full(40,z)],.005,6),'lower_grille' if z<.45 else 'front_grille')
    # Splitter: curved front lower aerodynamic lip.
    yy=np.linspace(-.87,.87,69);xx=-2.47+.20*(yy/.87)**2
    for z in [.211,.227]:b.mesh(tube(np.c_[xx,yy,np.full(len(yy),z)],.012,8),'splitter')
    # Integrated ducktail, swept and shallow (its shape is not a performance claim).
    R,U=np.meshgrid(np.linspace(-.88,.88,75),np.linspace(0,1,7),indexing='ij')
    X=2.045+.17*U-.075*R**2;Y=R;Z=skin(X,Y/width(X))[...,2]+.013+.04*U
    if design_skin:
        grid=np.stack((X,Y,Z),-1)
        # Authored 6-mm closed lip, not generic hole filling on an imported mesh.
        b.patch(grid,'spoiler'); low=grid.copy();low[:,:,2]-=.006
        b.patch(low,'spoiler',flip=True)
        for edge,edge_low in [(grid[0],low[0]),(grid[-1][::-1],low[-1][::-1]),
                              (grid[:,0][::-1],low[:,0][::-1]),(grid[:,-1],low[:,-1])]:
            b.patch(np.stack((edge,edge_low),axis=1),'spoiler')
    else: b.patch(np.stack((X,Y,Z),-1),'spoiler')
    b.mesh(tube(np.c_[X[:,-1],Y[:,-1],Z[:,-1]],.008,8),'spoiler')
    # Rear LED strip assemblies, black housings and connecting red accents.
    for s,suf in [(-1,'r'),(1,'l')]:
        yy=s*np.linspace(.15,.83,65);xx=2.442-.10*(yy/.862)**2
        zz=.869+.010*np.cos((np.abs(yy)-.15)*5)
        b.mesh(tube(np.c_[xx,yy,zz],.022,10),'taillamp_'+suf)
        line=np.c_[np.linspace(2.33,2.08,28),s*np.linspace(.871,.935,28),np.linspace(.875,.902,28)]
        b.mesh(tube(line,.016,8),'taillamp_'+suf)
    if design_skin:
        # One continuous authored body shell. It shares the exact side/end rim vertices.
        # A wheel-house ceiling is high over the tyre band, then blends into the floor.
        X,R=np.meshgrid(xs,rs,indexing='ij')
        ar=np.abs(R)
        t=np.clip((ar-.54)/.20,0,1); w=t*t*(3-2*t)
        rt=np.clip((ar-.65)/.20,0,1); rw=1-rt*rt*(3-2*rt)
        rise=.10*np.clip((X-1.8)/.63,0,1)**1.3*rw
        Z=.225+(lower(X)-.225)*w+rise*(1-w)
        # At the rear end, the profile must match end_surface exactly.
        Z[-1,:]=.225+.10*rw[-1,:]
        Y=(width(X)-.033)*R
        grid=np.stack((xe(X,R),Y,Z),axis=-1)
        xm=(X[:-1,:-1]+X[1:,1:])*.5; rm=(R[:-1,:-1]+R[1:,1:])*.5
        labs=np.full(xm.shape,b.ids['underbody'],np.int32)
        labs[(xm>1.8)&(np.abs(rm)<.8)]=b.ids['diffuser']
        for wx,ax in [(-1.49,'f'),(1.43,'b')]:
            for sign,suf in [(-1,'r'),(1,'l')]:
                labs[(np.abs(xm-wx)<.405)&(rm*sign>.54)]=b.ids[f'liner_{ax}{suf}']
        b.patch(grid,labs,flip=True)
    else:
        # Shallow undertray and a deliberately simplified diffuser insert.
        X,Y=np.meshgrid(np.linspace(-2.21,2.20,61),np.linspace(-.69,.69,29),indexing='ij')
        Z=np.full_like(X,.228);Z+=.095*np.clip((X-1.68)/.52,0,1)**2
        b.patch(np.stack((X,Y,Z),-1),'underbody',flip=True)
        X,Y=np.meshgrid(np.linspace(1.90,2.43,22),np.linspace(-.69,.69,33),indexing='ij');Z=.238+.19*((X-1.9)/.53)**1.3
        b.patch(np.stack((X,Y,Z),-1),'diffuser',flip=True)
        for y in np.linspace(-.57,.57,5):
            path=np.c_[np.linspace(1.91,2.43,22),np.full(22,y),np.linspace(.218,.395,22)]
            lower=path.copy();upper=path.copy();upper[:,2]+=.055
            b.patch(np.stack((lower,upper),axis=1),'diffuser')
    

    # Four independent wheel assemblies. Detailed rim spokes do not inflate semantic part count.
    for wx,ax in [(-1.49,'f'),(1.43,'b')]:
        for s,suf in [(-1,'r'),(1,'l')]:
            cy=s*.887;cz=.355
            prof=[[-.115,.253],[-.119,.286],[-.112,.321],[-.096,.344],[-.068,.355],
                  [0,.355],[.068,.355],[.096,.344],[.112,.321],[.119,.286],[.115,.253],[-.115,.253]]
            b.mesh(radial_surface(wx,cy,cz,prof,s),f'tire_{ax}{suf}')
            # Alloy outer lip and inner barrel.
            prof=[[.100,.249],[.123,.249],[.127,.241],[.122,.232],[.072,.227],[-.07,.229],[-.075,.244],[.10,.249]]
            b.mesh(radial_surface(wx,cy,cz,prof,s),f'rim_{ax}{suf}')
            b.mesh(radial_surface(wx,cy,cz,[[.052,.062],[.061,.062],[.061,.209],[.052,.209],[.052,.062]],s),f'brake_{ax}{suf}')
            # Ten directional sculpted spokes, paired on the opposite side by reflection.
            for theta in np.arange(10)*2*np.pi/10:
                # Quad cross-sections swept from hub to outer lip; subtle turbine twist.
                rr=np.linspace(.046,.239,9);tw=theta+.14*(rr-.046)/.193
                widths=.014-.006*(rr-.046)/.193
                vv=[]
                for radial,ang,half in zip(rr,tw,widths):
                    tang=np.array([-np.sin(ang),np.cos(ang)]);ctr=np.array([wx+radial*np.cos(ang),cz+radial*np.sin(ang)])
                    out=.097+.019*((radial-.046)/.193)**2
                    for dep,sgn in [(out,-1),(out,1),(out-.017,1),(out-.017,-1)]:
                        q=ctr+sgn*half*tang;vv.append([q[0],cy+s*dep,q[1]])
                ff=[]
                for i in range(len(rr)-1):
                    for j in range(4):a=4*i+j;c=4*(i+1)+j;d=4*(i+1)+(j+1)%4;bb=4*i+(j+1)%4;ff.extend([[a,bb,c],[bb,d,c]])
                ff.extend([[0,2,1],[0,3,2],[32,33,34],[32,34,35]])
                m=trimesh.Trimesh(vv,ff,process=False);m.fix_normals();b.mesh(m,f'rim_{ax}{suf}')
            prof=[[.096,0],[.103,.039],[.099,.049],[.082,.046],[.082,0]]
            b.mesh(radial_surface(wx,cy,cz,prof,s,64),f'rim_{ax}{suf}')
            for theta in np.arange(5)*2*np.pi/5:
                m=trimesh.creation.icosphere(subdivisions=1);m.vertices*=.007
                m.apply_translation([wx+.058*np.cos(theta),cy+s*.106,cz+.058*np.sin(theta)])
                b.mesh(m,f'rim_{ax}{suf}')

    # Geometric seals on the bonnet, windshield and rear glass boundaries.
    for sign in [-1,1]:
        xx=np.linspace(-2.055,-1.06,65);line=skin(xx,np.full(65,sign*.765));line[:,2]+=.0015
        b.mesh(tube(line,.0024,6),'hood_trim')
    for x in [-1.06,-.40,.94,1.62]:
        rr=np.linspace(-.77,.77,75);line=skin(np.full(75,x),rr);line[:,2]+=.0015
        b.mesh(tube(line,.0028,6),'hood_trim' if x==-1.06 else 'rear_trim' if x==1.62 else 'roof_trim')
    meta=dict(name='AeroGT Sportback · 高细节概念车',family_id='aerogt-sportback-v02',
        source='Original procedural multi-panel concept vehicle',asset_license='CC0-1.0',
        annotation_status='explicit_designed_exterior_modules_not_OEM_parts',
        reference_type='original_concept_not_real_production_vehicle',
        design_sets={
            'roof':['roof','panorama','pillar_l','pillar_r','roof_trim'],
            'rear':['tailgate','rear_glass','spoiler','rear_quarter_l','rear_quarter_r','taillamp_l','taillamp_r','rear_trim','roof_trim'],
            'bonnet':['hood','hood_trim'],
            'mirrors':['mirror_l','mirror_r','mirror_glass_l','mirror_glass_r'],
            'underbody':['diffuser','underbody']},
        notes=['No seats, engine, battery or unseen chassis detail.',
               'Exterior appearance and editable regions, not OEM dimension or CFD fidelity.',
               'Glass is an opaque visualization material; no optical transmission simulation.',
               'Panel/lighting/detail surfaces and wheel wells need independent CFD shell extraction.',
               'Do not interpret the default geometry as a closed fluid boundary.'],
        construction='Structured skin with wheel openings, tagged surfaces and geometrical trim; deterministic.')
    result=b.finish(meta)
    if design_skin:
        # Correct winding after constructing closed side walls; no vertex smoothing.
        mesh=result.mesh();mesh.fix_normals(multibody=True)
        result=result.clone(faces=np.asarray(mesh.faces,dtype=np.int32))
    return result

@lru_cache(maxsize=1)
def _reference(): return _build()

def detailed_model() -> Model:
    """Return an independent copy, so edits never mutate the cached reference."""
    return _reference().clone()
