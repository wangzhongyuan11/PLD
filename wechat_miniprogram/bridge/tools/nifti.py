# -*- coding: utf-8 -*-
"""Minimal NIfTI-1 single-file writer (stdlib only, gzipped .nii.gz).

Only what the ACDC demo cases need: float32 3D volumes with an sform affine
carrying the voxel spacing. Round-trip verified by nibabel-based consumers
elsewhere in the PLD workspace tests (acdc tests already accept synthetic
float32 volumes with a 2 mm isotropic affine).
"""
import gzip
import math
import struct


def write_nii_gz(path, volume, spacing=(2.0, 2.0, 2.0)):
    """volume: sequence of rows; each row is a sequence of d-lists:
    volume[x][y][z] float, x fastest. Writes <path> as gzipped NIfTI-1."""
    nx = len(volume)
    ny = len(volume[0]) if nx else 0
    nz = len(volume[0][0]) if ny else 0
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("volume too small")

    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)                    # sizeof_hdr
    struct.pack_into("<8h", header, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    header[70] = 16                                           # datatype float32
    struct.pack_into("<H", header, 72, 32)                    # bitpix
    struct.pack_into("<8f", header, 76, -1.0,
                     float(spacing[0]), float(spacing[1]), float(spacing[2]),
                     1.0, 1.0, 1.0, 1.0)                      # pixdim; qfac -1
    struct.pack_into("<f", header, 108, 352.0)                # vox_offset
    struct.pack_into("<f", header, 112, 1.0)                  # scl_slope
    struct.pack_into("<f", header, 116, 0.0)                  # scl_inter
    header[123] = 2                                            # xyzt_units: mm
    header[252] = 0                                           # qform_code
    header[254] = 1                                           # sform_code
    struct.pack_into("<4f", header, 280, float(spacing[0]), 0, 0, 0)
    struct.pack_into("<4f", header, 296, 0, float(spacing[1]), 0, 0)
    struct.pack_into("<4f", header, 312, 0, 0, float(spacing[2]), 0)
    header[344:348] = b"n+1\x00"

    # NIfTI-1: dim[1] (x) varies fastest in the file, so pack z-major.
    data = bytearray(nx * ny * nz * 4)
    offset = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                struct.pack_into("<f", data, offset, volume[x][y][z])
                offset += 4

    payload = bytes(header) + b"\x00\x00\x00\x00" + bytes(data)
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as fh:
        fh.write(payload)


def cardiac_phantom(size=(72, 72, 16), scale=1.0, seed=7):
    """Deterministic synthetic heart-like volume: bright LV cavity inside a
    myocardium shell over a faint background, values in [0, 1]."""
    nx, ny, nz = size
    state = seed

    def rnd():
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7fffffff
        return state / 0x7fffffff

    cx, cy = nx / 2.0, ny / 2.0
    r_lv = 0.22 * min(nx, ny) * scale
    r_my = 0.34 * min(nx, ny) * scale
    rz = nz / 2.0
    volume = []
    for x in range(nx):
        plane = []
        for y in range(ny):
            col = []
            for z in range(nz):
                dist = math.sqrt(((x - cx) / r_lv) ** 2 +
                                 ((y - cy) / r_lv) ** 2 +
                                 ((z - rz) / rz * 0.9) ** 2)
                dist_my = math.sqrt(((x - cx) / r_my) ** 2 +
                                     ((y - cy) / r_my) ** 2 +
                                     ((z - rz) / rz * 0.9) ** 2)
                if dist <= 1.0:
                    value = 0.92
                elif dist_my <= 1.0:
                    value = 0.55
                else:
                    value = 0.06 + 0.05 * rnd()
                col.append(min(1.0, max(0.0, value)))
            plane.append(col)
        volume.append(plane)
    return volume
