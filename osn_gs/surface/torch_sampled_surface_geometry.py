from __future__ import annotations
"""Deterministic sampled, piecewise-linear surface geometry helpers.

These are deliberately not an analytic NURBS/CAD intersection proof.
"""
from dataclasses import dataclass
from typing import Any
import math
import torch
from osn_gs.surface.torch_aabb_broad_phase import sweep_and_prune_pairs

@dataclass(frozen=True)
class SampledTriangleMesh:
    vertices: Any
    triangles: Any
    triangle_cells: Any
    resolution_u: int
    resolution_v: int
    diagonal_convention: str = "uv00_uv11"
    def triangle_points(self, index: int): return self.vertices[self.triangles[index]]
    def aabbs(self):
        p=self.vertices[self.triangles]
        return p.min(1).values, p.max(1).values
    def payload(self):
        return {"sample_resolution_u":self.resolution_u,"sample_resolution_v":self.resolution_v,"triangle_count":int(self.triangles.shape[0]),"diagonal_convention":self.diagonal_convention,"dtype":str(self.vertices.dtype),"device":str(self.vertices.device)}

def sample_nurbs_triangle_mesh(surface: Any, resolution_u: int=16, resolution_v: int=16) -> SampledTriangleMesh:
    if resolution_u < 2 or resolution_v < 2: raise ValueError("sample resolution must be >= 2")
    device, dtype=surface.control_grid.device, surface.control_grid.dtype
    u=torch.linspace(0.,1.,resolution_u,dtype=dtype,device=device); v=torch.linspace(0.,1.,resolution_v,dtype=dtype,device=device)
    uu,vv=torch.meshgrid(u,v,indexing="ij"); vertices=surface.evaluate(torch.stack([uu.reshape(-1),vv.reshape(-1)],1)).detach().cpu().to(torch.float64).contiguous()
    tri=[]; cells=[]
    for i in range(resolution_u-1):
      for j in range(resolution_v-1):
       a=i*resolution_v+j; b=(i+1)*resolution_v+j; c=(i+1)*resolution_v+j+1; d=i*resolution_v+j+1
       tri.extend(((a,b,c),(a,c,d))); cells.extend(((i,j),(i,j)))
    return SampledTriangleMesh(vertices,torch.tensor(tri,dtype=torch.long),torch.tensor(cells,dtype=torch.long),resolution_u,resolution_v)

def build_triangle_aabb_pairs(mesh_a: SampledTriangleMesh, mesh_b: SampledTriangleMesh|None=None, *, tolerance: float=1e-9):
    same=mesh_b is None; mesh_b=mesh_a if same else mesh_b
    amin,amax=mesh_a.aabbs(); bmin,bmax=mesh_b.aabbs(); mins=torch.cat([amin,bmin]); maxs=torch.cat([amax,bmax]); labels=[f"a:{i}" for i in range(len(amin))]+[f"b:{i}" for i in range(len(bmin))]
    pairs=sweep_and_prune_pairs(labels,mins,maxs,[float(tolerance)]*len(labels),expand_factor=1.0,tol=float(tolerance))
    out=[]
    for p in pairs:
      la,lb=p.label_a,p.label_b
      if same:
       ia=int(la.split(':')[1]); ib=int(lb.split(':')[1]);
       if ia<ib: out.append((ia,ib))
      elif la[0]!=lb[0]: out.append((int(la.split(':')[1]),int(lb.split(':')[1])) if la[0]=='a' else (int(lb.split(':')[1]),int(la.split(':')[1])))
    return out

def _v(x): return [float(q) for q in x]
def _sub(a,b): return [a[i]-b[i] for i in range(3)]
def _dot(a,b): return sum(x*y for x,y in zip(a,b))
def _cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def _norm(a): return math.sqrt(_dot(a,a))
def _point_tri(p,t,eps):
 a,b,c=map(_v,t); p=_v(p); n=_cross(_sub(b,a),_sub(c,a)); nn=_norm(n)
 if nn<=eps or abs(_dot(n,_sub(p,a)))>eps*nn:return False
 v0=_sub(c,a);v1=_sub(b,a);v2=_sub(p,a); den=_dot(v0,v0)*_dot(v1,v1)-_dot(v0,v1)**2
 if abs(den)<=eps:return False
 u=(_dot(v1,v1)*_dot(v2,v0)-_dot(v0,v1)*_dot(v2,v1))/den; v=(_dot(v0,v0)*_dot(v2,v1)-_dot(v0,v1)*_dot(v2,v0))/den
 return u>=-eps and v>=-eps and u+v<=1+eps

def triangle_triangle_intersection(tri_a: Any, tri_b: Any, *, tolerance: float=1e-9) -> dict[str,Any]:
 """Tolerance-aware sampled triangle contact classification; deterministic, not CAD exact."""
 a,b=[[_v(x) for x in tri] for tri in (tri_a,tri_b)]; na=_cross(_sub(a[1],a[0]),_sub(a[2],a[0])); nb=_cross(_sub(b[1],b[0]),_sub(b[2],b[0])); nna, nnb=_norm(na),_norm(nb)
 if nna<=tolerance or nnb<=tolerance:return {"intersects":False,"kind":"degenerate"}
 da=[_dot(na,_sub(p,a[0]))/nna for p in b]; db=[_dot(nb,_sub(p,b[0]))/nnb for p in a]
 cop=max(map(abs,da))<=1e-12 and max(map(abs,db))<=1e-12  # exact coplanarity; tolerance-near planes are not crossings
 # Segment-plane crossings plus endpoint containment covers non-coplanar triangles.
 hits=[]
 for p,q in ((a[i],a[(i+1)%3]) for i in range(3)):
  d0=_dot(nb,_sub(p,b[0]))/nnb; d1=_dot(nb,_sub(q,b[0]))/nnb
  if d0*d1<=0 and abs(d0-d1)>1e-15:
   r=d0/(d0-d1); x=[p[k]+r*(q[k]-p[k]) for k in range(3)]
   if _point_tri(x,b,tolerance): hits.append(x)
 for p,q in ((b[i],b[(i+1)%3]) for i in range(3)):
  d0=_dot(na,_sub(p,a[0]))/nna; d1=_dot(na,_sub(q,a[0]))/nna
  if d0*d1<=0 and abs(d0-d1)>1e-15:
   r=d0/(d0-d1); x=[p[k]+r*(q[k]-p[k]) for k in range(3)]
   if _point_tri(x,a,tolerance): hits.append(x)
 if hits:return {"intersects":True,"kind":"proper_interior_intersection","points":sorted(hits)}
 if cop:
  # coplanar: vertex containment is sufficient for the sampled fixtures and reports contact conservatively.
  inside=[p for p in a if _point_tri(p,b,tolerance)]+[p for p in b if _point_tri(p,a,tolerance)]
  if inside:return {"intersects":True,"kind":"coplanar_area_overlap","points":sorted(inside)}
 return {"intersects":False,"kind":"none"}