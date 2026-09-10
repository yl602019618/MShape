"""BVH triangle queries in metres. Floating-point tolerance, not exact predicates.

No vertex-sampling-only clearance test: minimum distance traverses triangle BVHs
and includes point/triangle and edge/edge features. Intersections include
coplanar overlap; a mesh's expected shared edge/vertex contact is excluded,
NOT every adjacent triangle pair. Query budgets fail closed (incomplete != clear).
"""
from __future__ import annotations
import time
import numpy as np
from numba import njit


@njit(cache=True)
def _dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
@njit(cache=True)
def _cross(a,b): return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
@njit(cache=True)
def _point_tri(p,t):
    a,b,c=t[0],t[1],t[2];ab=b-a;ac=c-a;ap=p-a
    d1=_dot(ab,ap);d2=_dot(ac,ap)
    if d1<=0 and d2<=0:return a.copy(),np.array([1.,0.,0.])
    bp=p-b;d3=_dot(ab,bp);d4=_dot(ac,bp)
    if d3>=0 and d4<=d3:return b.copy(),np.array([0.,1.,0.])
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        s=d1/max(d1-d3,1e-300);return a+s*ab,np.array([1-s,s,0.])
    cp=p-c;d5=_dot(ab,cp);d6=_dot(ac,cp)
    if d6>=0 and d5<=d6:return c.copy(),np.array([0.,0.,1.])
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        s=d2/max(d2-d6,1e-300);return a+s*ac,np.array([1-s,0.,s])
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:
        s=(d4-d3)/max(d4-d3+d5-d6,1e-300);return b+s*(c-b),np.array([0.,1-s,s])
    den=va+vb+vc
    if abs(den)<1e-300:return a.copy(),np.array([1.,0.,0.])
    v=vb/den;w=vc/den;return a+v*ab+w*ac,np.array([1-v-w,v,w])

@njit(cache=True)
def _seg_seg(p1,q1,p2,q2):
    d1=q1-p1;d2=q2-p2;r=p1-p2;a=_dot(d1,d1);e=_dot(d2,d2);f=_dot(d2,r)
    if a<=1e-28 and e<=1e-28:return p1.copy(),p2.copy()
    if a<=1e-28:s=0.;t=min(1.,max(0.,f/e))
    else:
        c=_dot(d1,r)
        if e<=1e-28:t=0.;s=min(1.,max(0.,-c/a))
        else:
            b=_dot(d1,d2);den=a*e-b*b
            s=min(1.,max(0.,(b*f-c*e)/den)) if den>1e-28 else 0.
            t=(b*s+f)/e
            if t<0:t=0.;s=min(1.,max(0.,-c/a))
            elif t>1:t=1.;s=min(1.,max(0.,(b-c)/a))
    return p1+s*d1,p2+t*d2

@njit(cache=True)
def _expected(p,a,ia,ib,tol):
    common=np.empty(3,np.int64);n=0
    for i in range(3):
        for j in range(3):
            if ia[i]==ib[j]:common[n]=i;n+=1;break
    if n==0 or n==3:return False
    if n==1:
        d=p-a[common[0]];return _dot(d,d)<=tol*tol*16
    x=a[common[0]];y=a[common[1]];d=y-x
    t=min(1.,max(0.,_dot(p-x,d)/max(_dot(d,d),1e-300)))
    e=p-x-t*d;return _dot(e,e)<=tol*tol*16

@njit(cache=True)
def _segment_triangle(p,q,t,tol):
    direction=q-p;e1=t[1]-t[0];e2=t[2]-t[0];h=_cross(direction,e2);det=_dot(e1,h)
    scale=np.sqrt(_dot(direction,direction)*_dot(e1,e1)*_dot(e2,e2))
    if abs(det)<=max(scale*1e-13,1e-30):return False,p
    inv=1/det;s=p-t[0];u=_dot(s,h)*inv
    e=tol/max(np.sqrt(_dot(e1,e1)),np.sqrt(_dot(e2,e2)),1e-12)
    if u< -e or u>1+e:return False,p
    h2=_cross(s,e1);v=_dot(direction,h2)*inv
    if v< -e or u+v>1+e:return False,p
    x=_dot(e2,h2)*inv
    et=tol/max(np.sqrt(_dot(direction,direction)),1e-12)
    if x< -et or x>1+et:return False,p
    return True,p+min(1.,max(0.,x))*direction

@njit(cache=True)
def _cross2(a,b):return a[0]*b[1]-a[1]*b[0]

@njit(cache=True)
def _coplanar_overlap(a,b,normal,tol):
    # Clip a by b in the best-conditioned 2-D projection. Area, not a contact point.
    drop=int(np.argmax(np.abs(normal)));axes=np.empty(2,np.int64);k=0
    for i in range(3):
        if i!=drop:axes[k]=i;k+=1
    p=np.zeros((12,2));q=np.zeros((3,2))
    for i in range(3):
        for j in range(2):p[i,j]=a[i,axes[j]];q[i,j]=b[i,axes[j]]
    sign=1. if _cross2(q[1]-q[0],q[2]-q[0])>=0 else -1.;n=3
    for j in range(3):
        if n==0:return False
        out=np.zeros((12,2));m=0;u=q[j];v=q[(j+1)%3];edge=v-u
        for i in range(n):
            s=p[i];t=p[(i+1)%n];ds=sign*_cross2(edge,s-u);dt=sign*_cross2(edge,t-u)
            ins=ds>=0;intt=dt>=0
            if ins:
                out[m]=s;m+=1
            if ins!=intt:
                out[m]=s+(t-s)*(ds/(ds-dt));m+=1
        p=out;n=m
    area=0.
    for i in range(n):area+=_cross2(p[i],p[(i+1)%n])
    return abs(area)*.5>tol*tol*4

@njit(cache=True)
def _tri_intersects(a,b,ia,ib,tol):
    na=_cross(a[1]-a[0],a[2]-a[0]);nb=_cross(b[1]-b[0],b[2]-b[0])
    la=np.sqrt(_dot(na,na));lb=np.sqrt(_dot(nb,nb))
    if la<1e-20 or lb<1e-20:return False
    da=np.empty(3);db=np.empty(3)
    for i in range(3):da[i]=_dot(b[i]-a[0],na)/la;db[i]=_dot(a[i]-b[0],nb)/lb
    if np.min(da)>tol or np.max(da)<-tol or np.min(db)>tol or np.max(db)<-tol:return False
    if np.max(np.abs(da))<=tol and np.max(np.abs(db))<=tol:
        if _coplanar_overlap(a,b,na,tol):return True
        # Non-adjacent coplanar contact still matters for assembly diagnosis.
        common=False
        for i in range(3):
            for j in range(3):
                if ia[i]==ib[j]:common=True
        if common:return False
        for i in range(3):
            for j in range(3):
                p,q=_seg_seg(a[i],a[(i+1)%3],b[j],b[(j+1)%3]);d=p-q
                if _dot(d,d)<=tol*tol:return True
        return False
    # Analytic shared-simplex classification prevents ill-conditioned, nearly
    # in-plane edge/triangle solves from inventing tiny crossings at a common
    # vertex. This is NOT blanket adjacency exclusion: a shared-vertex pair
    # whose two wedges straddle both planes still reaches the intersection test.
    shared_a=np.zeros(3,np.bool_);shared_b=np.zeros(3,np.bool_);common=0
    for i in range(3):
        for j in range(3):
            if ia[i]==ib[j]:shared_a[i]=True;shared_b[j]=True;common+=1
    if common==2:return False  # distinct planes meet only along their shared edge
    if common==1:
        apos=True;aneg=True;bpos=True;bneg=True
        for i in range(3):
            if not shared_a[i]:apos=apos and db[i]>tol;aneg=aneg and db[i]<-tol
            if not shared_b[i]:bpos=bpos and da[i]>tol;bneg=bneg and da[i]<-tol
        if apos or aneg or bpos or bneg:return False
    for i in range(3):
        hit,p=_segment_triangle(a[i],a[(i+1)%3],b,tol)
        if hit and not _expected(p,a,ia,ib,tol):return True
        hit,p=_segment_triangle(b[i],b[(i+1)%3],a,tol)
        if hit and not _expected(p,a,ia,ib,tol):return True
    return False

@njit(cache=True)
def _tri_distance(a,b):
    # Crossing interiors need explicit intersection detection; edge-edge alone is insufficient.
    for i in range(3):
        h,p=_segment_triangle(a[i],a[(i+1)%3],b,1e-12)
        if h:return 0.,p,p
        h,p=_segment_triangle(b[i],b[(i+1)%3],a,1e-12)
        if h:return 0.,p,p
    best=1e300;pa=a[0].copy();pb=b[0].copy()
    for i in range(3):
        p,_=_point_tri(a[i],b);d=p-a[i];dd=_dot(d,d)
        if dd<best:best=dd;pa=a[i].copy();pb=p
        p,_=_point_tri(b[i],a);d=p-b[i];dd=_dot(d,d)
        if dd<best:best=dd;pa=p;pb=b[i].copy()
        for j in range(3):
            p,q=_seg_seg(a[i],a[(i+1)%3],b[j],b[(j+1)%3]);d=p-q;dd=_dot(d,d)
            if dd<best:best=dd;pa=p;pb=q
    return best,pa,pb

@njit(cache=True)
def _box_dist(lo,hi,lo2,hi2):
    d=0.
    for k in range(3):
        x=max(0.,lo[k]-hi2[k],lo2[k]-hi[k]);d+=x*x
    return d


def build_bvh(vertices,faces,leaf=8):
    triangles=np.ascontiguousarray(vertices[faces],dtype=np.float64)
    lo=triangles.min(1);hi=triangles.max(1);cent=(lo+hi)*.5
    order=np.arange(len(faces),dtype=np.int64);nodes=[]
    def rec(start,end):
        idx=len(nodes);ids=order[start:end];bounds=(lo[ids].min(0),hi[ids].max(0))
        nodes.append([*bounds,-1,-1,start,end])
        if end-start>leaf:
            axis=int(np.ptp(cent[ids],axis=0).argmax());mid=(end+start)//2
            order[start:end]=ids[np.argpartition(cent[ids,axis],mid-start)]
            nodes[idx][2]=rec(start,mid);nodes[idx][3]=rec(mid,end)
        return idx
    if not len(faces):raise ValueError('Empty triangle set for spatial query.')
    rec(0,len(faces))
    return (triangles,np.array([n[0] for n in nodes]),np.array([n[1] for n in nodes]),
            np.array([n[2:] for n in nodes],np.int64),order)

@njit(cache=True)
def _self_query(v,f,tri,lo,hi,nodes,order,tol,pair_limit,test_limit):
    # Per-face BVH traversal produces each unordered candidate once.
    result=np.empty((pair_limit,2),np.int64);found=0;tests=0;stack=np.empty(128,np.int64)
    for face in range(len(f)):
        amin,amax=_bounds(tri[face])
        size=1;stack[0]=0
        while size:
            size-=1;node=stack[size]
            if _box_dist(amin,amax,lo[node],hi[node])>tol*tol:continue
            left,right,start,end=nodes[node]
            if left>=0:
                stack[size]=left;stack[size+1]=right;size+=2;continue
            for k in range(start,end):
                j=order[k]
                if j<=face:continue
                tests+=1
                if tests>test_limit:return result[:found],False,tests
                if _tri_intersects(tri[face],tri[j],f[face],f[j],tol):
                    result[found,0]=face;result[found,1]=j;found+=1
                    if found>=pair_limit:return result[:found],False,tests
    return result[:found],True,tests

# NumPy's axis reduction on 2D arrays is not supported in every Numba release.
@njit(cache=True)
def _bounds(t):
    a=t[0].copy();b=t[0].copy()
    for i in range(1,3):
        for j in range(3):a[j]=min(a[j],t[i,j]);b[j]=max(b[j],t[i,j])
    return a,b

@njit(cache=True)
def _nearest_points(points,tri,lo,hi,nodes,order):
    xyz=np.empty_like(points);face=np.empty(len(points),np.int64);bary=np.empty_like(points)
    stack=np.empty(128,np.int64)
    for i in range(len(points)):
        p=points[i];best=1e300;size=1;stack[0]=0;face[i]=-1
        while size:
            size-=1;node=stack[size]
            if _box_dist(p,p,lo[node],hi[node])>best:continue
            left,right,start,end=nodes[node]
            if left>=0:
                d1=_box_dist(p,p,lo[left],hi[left]);d2=_box_dist(p,p,lo[right],hi[right])
                stack[size]=left if d1>d2 else right;stack[size+1]=right if d1>d2 else left;size+=2
            else:
                for k in range(start,end):
                    j=order[k];q,b=_point_tri(p,tri[j]);delta=p-q;d=_dot(delta,delta)
                    if d<best:best=d;xyz[i]=q;bary[i]=b;face[i]=j
    return xyz,face,bary

@njit(cache=True)
def _distance_query(ta,loa,hia,na,oa,tb,lob,hib,nb,ob,test_limit):
    best=1e300;pa=np.zeros(3);pb=np.zeros(3);fa=-1;fb=-1;tests=0
    sa=np.empty(256,np.int64);sb=np.empty(256,np.int64);sa[0]=0;sb[0]=0;n=1
    while n:
        n-=1;a=sa[n];b=sb[n]
        if _box_dist(loa[a],hia[a],lob[b],hib[b])>best:continue
        al,ar,astart,aend=na[a];bl,br,bstart,bend=nb[b]
        if al<0 and bl<0:
            for i in range(astart,aend):
                for j in range(bstart,bend):
                    tests+=1
                    if tests>test_limit:return best,pa,pb,fa,fb,False,tests
                    d,p,q=_tri_distance(ta[oa[i]],tb[ob[j]])
                    if d<best:best=d;pa=p;pb=q;fa=oa[i];fb=ob[j]
                    if best<=1e-24:return 0.,pa,pb,fa,fb,True,tests
        else:
            if bl<0 or (al>=0 and aend-astart>=bend-bstart):
                d1=_box_dist(loa[al],hia[al],lob[b],hib[b]);d2=_box_dist(loa[ar],hia[ar],lob[b],hib[b])
                sa[n]=al if d1>d2 else ar;sa[n+1]=ar if d1>d2 else al;sb[n]=b;sb[n+1]=b
            else:
                d1=_box_dist(loa[a],hia[a],lob[bl],hib[bl]);d2=_box_dist(loa[a],hia[a],lob[br],hib[br])
                sb[n]=bl if d1>d2 else br;sb[n+1]=br if d1>d2 else bl;sa[n]=a;sa[n+1]=a
            n+=2
    return best,pa,pb,fa,fb,True,tests

class TriangleIndex:
    def __init__(self,vertices,faces):
        self.vertices=np.ascontiguousarray(vertices,np.float64);self.faces=np.ascontiguousarray(faces,np.int64)
        self.bvh=build_bvh(self.vertices,self.faces)
    def closest(self,points):return _nearest_points(np.ascontiguousarray(points,np.float64),*self.bvh)
    def intersections(self,*,tolerance=1e-8,max_pairs=20000,max_tests=20_000_000):
        start=time.perf_counter();pairs,complete,tests=_self_query(self.vertices,self.faces,*self.bvh,tolerance,max_pairs,max_tests)
        return dict(pairs=pairs.tolist(),complete=bool(complete),triangle_tests=int(tests),
                    count=len(pairs),tolerance_m=tolerance,elapsed_s=time.perf_counter()-start,
                    status=('intersections_found' if len(pairs) else 'clear' if complete else 'incomplete'),
                    numerical_method='float64_BVH_triangle_tests_coplanar_overlap_not_exact_predicates')
    def distance(self,other,*,max_tests=5_000_000):
        d,p,q,a,b,complete,tests=_distance_query(*self.bvh,*other.bvh,max_tests)
        return dict(distance_m=float(np.sqrt(d)),point_a=p.tolist(),point_b=q.tolist(),
                    face_a=int(a),face_b=int(b),complete=bool(complete),triangle_tests=int(tests),
                    measurement='minimum_unsigned_triangle_surface_distance' if complete else 'upper_bound_only',
                    containment_checked=False)
