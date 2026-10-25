"""Standalone HDMI viewer for completed ACDC meshes. Stop all PL owners first.

Touch drag rotates; top-left switches ED/ES, top-middle resets, top-right exits.
No inference or quantitative computation is performed here.
"""
import argparse, json, math, os, select, signal, struct, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def _pack(points,triangles,colors):
    p=np.ascontiguousarray(np.concatenate(points),dtype=np.float32)
    p-=(p.min(0)+p.max(0))/2
    idx=np.ascontiguousarray(np.concatenate(triangles),dtype=np.uint32)
    faces=p[idx]
    normals=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9)
    return p,idx,np.asarray(colors,dtype=np.float32),max(float(np.linalg.norm(np.ptp(p,axis=0))),1),normals

def load_mesh(root, phase):
    volume=json.loads((root/'viewer'/phase/'volume.json').read_text())
    points=[];triangles=[];colors=[];offset=0
    preview_points=[];preview_triangles=[];preview_colors=[];preview_offset=0
    for row in volume['meshes']:
        meta=row['smooth'];folder=root/'meshes'/phase
        p=np.fromfile(folder/meta['points'],dtype='<f4').reshape(-1,3)
        idx=np.fromfile(folder/meta['indices'],dtype='<u4').reshape(-1,3)
        if len(p)!=meta['vertex_count'] or len(idx)!=meta['triangle_count'] or idx.max()>=len(p):
            raise ValueError('invalid mesh dimensions')
        points.append(p);triangles.append(idx+offset);offset+=len(p)
        colors.extend([row['color']]*len(idx))
        # Interaction-only vertex clustering; the full mesh remains unchanged.
        cell=max(float(np.linalg.norm(np.ptp(p,axis=0)))/28,1e-6)
        _,mapping=np.unique(np.floor((p-p.min(0))/cell).astype(np.int32),axis=0,return_inverse=True)
        counts=np.bincount(mapping)
        small=np.column_stack([np.bincount(mapping,weights=p[:,axis])/counts for axis in range(3)]).astype(np.float32)
        faces=mapping[idx]
        keep=(faces[:,0]!=faces[:,1])&(faces[:,0]!=faces[:,2])&(faces[:,1]!=faces[:,2])
        faces=faces[keep]
        _,unique=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
        faces=faces[np.sort(unique)]
        preview_points.append(small);preview_triangles.append(faces+preview_offset);preview_offset+=len(small)
        preview_colors.extend([row['color']]*len(faces))
    full=_pack(points,triangles,colors)
    preview=_pack(preview_points,preview_triangles,preview_colors)
    # Use the same world-space origin and scale in both representations.
    full_center=(np.concatenate(points).min(0)+np.concatenate(points).max(0))/2
    preview_raw=np.concatenate(preview_points)
    preview_center=(preview_raw.min(0)+preview_raw.max(0))/2
    preview[0][:]+=preview_center-full_center
    preview=(preview[0],preview[1],preview[2],full[3],preview[4])
    return full,preview

def render(mesh,yaw,pitch,phase,preview=False,size=(640,432)):
    p,idx,colors,diameter,normals=mesh[1 if preview else 0]
    a,b=math.cos(yaw),math.sin(yaw);c,d=math.cos(pitch),math.sin(pitch)
    rotation=np.array([[a,0,b],[b*d,c,-a*d],[-b*c,d,a*c]],dtype=np.float32)
    v=p@rotation.T
    light=.3+.7*np.abs(normals@rotation[2])
    shades=np.clip(colors*light[:,None],0,255).astype(np.uint8)
    width,height=size
    xy=v[:,:2]*(min(width,height)*.92/diameter);xy[:,0]+=width/2;xy[:,1]=height/2-xy[:,1]
    import ctypes
    library=ctypes.CDLL(str(Path(__file__).with_name('surface_raster.so')))
    projected=np.ascontiguousarray(np.column_stack((xy,v[:,2])),dtype=np.float32)
    indices=np.ascontiguousarray(idx,dtype=np.uint32);shades=np.ascontiguousarray(shades)
    pixels=np.empty((height,width,3),dtype=np.uint8)
    library.surface_raster.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    if library.surface_raster(projected.ctypes.data,indices.ctypes.data,shades.ctypes.data,len(idx),pixels.ctypes.data,width,height):
        raise MemoryError('surface depth buffer')
    return pixels
