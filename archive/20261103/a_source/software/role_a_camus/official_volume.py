"""CAMUS official notebook's 20-disk biplane method, with no fallback.

Upstream: Human Heart Project CAMUS_public/jupyter/LeftVentricleVolumes.ipynb.
Data/reference license: CC BY-NC-SA 4.0; cite Leclerc et al., IEEE TMI 2019.
Only named geometry functions were extracted; no data loading or demo executes.
The disk helper copies diameter arrays before unit conversion. The 2D cross
product is written explicitly for NumPy >= 2 without changing the formula.
"""
from __future__ import annotations
import logging, math
from typing import Any, Dict, Tuple, Mapping, Sequence
import numpy as np
import PIL
from PIL.Image import Resampling
from scipy import ndimage
from skimage.measure import find_contours
logger = logging.getLogger(__name__)
DISK_COUNT = 20
CITATION = "Leclerc et al., IEEE TMI 2019, DOI 10.1109/TMI.2019.2900516"
LICENSE = "CC BY-NC-SA 4.0, noncommercial scientific research"

REFERENCE_SOURCE_SHA256 = 'e947295d6ea1e25c27caf6281cf1b9c2568700e0cebda46c40b91d2d5ec68bfb'

def resize_image(image: np.ndarray, size: Tuple[int, int], resample: Resampling = Resampling.NEAREST) -> np.ndarray:
    """Resizes the image to the specified dimensions.

    Args:
        image: (H, W), Input image to resize. Must be in a format supported by PIL.
        size: Width (W') and height (H') dimensions of the resized image to output.
        resample: Resampling filter to use.

    Returns:
        (H', W'), Input image resized to the specified dimensions.
    """
    resized_image = np.array(PIL.Image.fromarray(image).resize(size, resample=resample))
    return resized_image

def resize_image_to_isotropic(
    image: np.ndarray, spacing: Tuple[float, float], resample: Resampling = Resampling.NEAREST
) -> np.ndarray:
    """Resizes the image to attain isotropic spacing, by resampling the dimension with the biggest voxel size.

    Args:
        image: (H, W), Input image to resize. Must be in a format supported by PIL.
        spacing: Size of the image's pixels along each (height, width) dimension.
        resample: Resampling filter to use.

    Returns:
        (H', W'), Input image resized so that the spacing is isotropic, and the isotropic value of the new spacing.
    """
    scaling = np.array(spacing) / min(spacing)
    new_height, new_width = (np.array(image.shape) * scaling).round().astype(int)
    return resize_image(image, (new_width, new_height), resample=resample), min(spacing)

def compute_left_ventricle_volumes(
    a2c_ed: np.ndarray,
    a2c_es: np.ndarray,
    a2c_voxelspacing: Tuple[float, float],
    a4c_ed: np.ndarray,
    a4c_es: np.ndarray,
    a4c_voxelspacing: Tuple[float, float],
) -> Tuple[float, float]:
    """Computes the ED and ES volumes of the left ventricle from 2 orthogonal 2D views (A2C and A4C).

    Args:
        a2c_ed: (H,W), Binary segmentation map of the left ventricle from the end-diastole (ED) instant of the 2-chamber
            apical view (A2C).
        a2c_es: (H,W), Binary segmentation map of the left ventricle from the end-systole (ES) instant of the 2-chamber
            apical view (A2C).
        a2c_voxelspacing: Size (in mm) of the 2-chamber apical view's voxels along each (height, width) dimension.
        a4c_ed: (H,W), Binary segmentation map of the left ventricle from the end-diastole (ED) instant of the 4-chamber
            apical view (A4C).
        a4c_es: (H,W), Binary segmentation map of the left ventricle from the end-systole (ES) instant of the 4-chamber
            apical view (A4C).
        a4c_voxelspacing: Size (in mm) of the 4-chamber apical view's voxels along each (height, width) dimension.

    Returns:
        Left ventricle ED and ES volumes.
    """
    for mask_name, mask in [("a2c_ed", a2c_ed), ("a2c_es", a2c_es), ("a4c_ed", a4c_ed), ("a4c_es", a4c_es)]:
        if mask.max() > 1:
            logger.warning(
                f"`compute_left_ventricle_volumes` expects binary segmentation masks of the left ventricle (LV). "
                f"However, the `{mask_name}` segmentation contains a label greater than '1/True'. If this was done "
                f"voluntarily, you can safely ignore this warning. However, the most likely cause is that you forgot "
                f"to extract the binary LV segmentation from a multi-class segmentation mask."
            )

    a2c_ed_diameters, a2c_ed_step_size = _compute_diameters(a2c_ed, a2c_voxelspacing)
    a2c_es_diameters, a2c_es_step_size = _compute_diameters(a2c_es, a2c_voxelspacing)
    a4c_ed_diameters, a4c_ed_step_size = _compute_diameters(a4c_ed, a4c_voxelspacing)
    a4c_es_diameters, a4c_es_step_size = _compute_diameters(a4c_es, a4c_voxelspacing)
    step_size = max((a2c_ed_step_size, a2c_es_step_size, a4c_ed_step_size, a4c_es_step_size))

    ed_volume = _compute_left_ventricle_volume_by_instant(a2c_ed_diameters, a4c_ed_diameters, step_size)
    es_volume = _compute_left_ventricle_volume_by_instant(a2c_es_diameters, a4c_es_diameters, step_size)
    return ed_volume, es_volume

def _compute_left_ventricle_volume_by_instant(
    a2c_diameters: np.ndarray, a4c_diameters: np.ndarray, step_size: float
) -> float:
    """Compute left ventricle volume using Biplane Simpson's method.

    Args:
        a2c_diameters: Diameters measured at each key instant of the cardiac cycle, from the 2-chamber apical view.
        a4c_diameters: Diameters measured at each key instant of the cardiac cycle, from the 4-chamber apical view.
        step_size:

    Returns:
        Left ventricle volume (in millilitres).
    """
    # All measures are now in millimeters, convert to meters by dividing by 1000
    a2c_diameters = np.array(a2c_diameters, dtype=float, copy=True) / 1000
    a4c_diameters = np.array(a4c_diameters, dtype=float, copy=True) / 1000
    step_size /= 1000

    # Estimate left ventricle volume from orthogonal disks
    lv_volume = np.sum(a2c_diameters * a4c_diameters) * step_size * np.pi / 4

    # Volume is now in cubic meters, so convert to milliliters (1 cubic meter = 1_000_000 milliliters)
    return round(lv_volume * 1e6)

def _find_distance_to_edge(
    segmentation: np.ndarray, point_on_mid_line: np.ndarray, normal_direction: np.ndarray
) -> float:
    distance = 8  # start a bit in to avoid line stopping early at base
    while True:
        current_position = point_on_mid_line + distance * normal_direction

        y, x = np.round(current_position).astype(int)
        if segmentation.shape[0] <= y or y < 0 or segmentation.shape[1] <= x or x < 0:
            # out of bounds
            return distance

        elif segmentation[y, x] == 0:
            # Edge found
            return distance

        distance += 0.5

def _distance_line_to_points(line_point_0, line_point_1, points):
    delta = line_point_1 - line_point_0
    rel = line_point_0 - points
    return np.abs(delta[0]*rel[:, 1] - delta[1]*rel[:, 0]) / np.linalg.norm(delta)


def _get_angle_of_lines_to_point(reference_point: np.ndarray, moving_points: np.ndarray) -> np.ndarray:
    diff = moving_points - reference_point
    return abs(np.degrees(np.arctan2(diff[:, 0], diff[:, 1])))

def _compute_diameters(segmentation: np.ndarray, voxelspacing: Tuple[float, float]) -> Tuple[np.ndarray, float]:
    """

    Args:
        segmentation: Binary segmentation of the structure for which to find the diameter.
        voxelspacing: Size of the segmentations' voxels along each (height, width) dimension (in mm).

    Returns:
    """

    # Make image isotropic, have same spacing in both directions.
    # The spacing can be multiplied by the diameter directly.
    segmentation, isotropic_spacing = resize_image_to_isotropic(segmentation, voxelspacing)

    # Go through entire contour to find AV plane
    contour = find_contours(segmentation, 0.5)[0]

    # For each pair of contour points
    # Check if angle is ok
    # If angle is ok, check that almost all other contour points are above the line
    # Or check that all points between are close to the line
    # If so, it is accepted, select the longest stretch
    best_length = 0
    for point_idx in range(2, len(contour)):
        previous_points = contour[:point_idx]
        angles_to_previous_points = _get_angle_of_lines_to_point(contour[point_idx], previous_points)

        for acute_angle_idx in np.nonzero(angles_to_previous_points <= 45)[0]:
            intermediate_points = contour[acute_angle_idx + 1 : point_idx]
            distance_to_intermediate_points = _distance_line_to_points(
                contour[point_idx], contour[acute_angle_idx], intermediate_points
            )
            if np.all(distance_to_intermediate_points <= 8):
                distance = np.linalg.norm(contour[point_idx] - contour[acute_angle_idx])
                if best_length < distance:
                    best_length = distance
                    best_i = point_idx
                    best_j = acute_angle_idx

    mid_point = int(best_j + round((best_i - best_j) / 2))
    # Apex is longest from midpoint
    mid_line_length = 0
    apex = 0
    for i in range(len(contour)):
        length = np.linalg.norm(contour[mid_point] - contour[i])
        if mid_line_length < length:
            mid_line_length = length
            apex = i

    direction = contour[apex] - contour[mid_point]
    normal_direction = np.array([-direction[1], direction[0]])
    normal_direction = normal_direction / np.linalg.norm(normal_direction)  # Normalize
    diameters = []
    for fraction in np.linspace(0, 1, 20, endpoint=False):
        point_on_mid_line = contour[mid_point] + direction * fraction

        distance1 = _find_distance_to_edge(segmentation, point_on_mid_line, normal_direction)
        distance2 = _find_distance_to_edge(segmentation, point_on_mid_line, -normal_direction)
        diameters.append((distance1 + distance2) * isotropic_spacing)

    step_size = (mid_line_length * isotropic_spacing) / 20
    return np.array(diameters), step_size

def volumes_by_instant(a2c_diameters, a4c_diameters, step_size):
    """Exactly 20 valid disk diameters in mm; return official rounded mL."""
    a=np.asarray(a2c_diameters); b=np.asarray(a4c_diameters)
    if a.shape!=(20,) or b.shape!=(20,) or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any() or not math.isfinite(step_size) or step_size<=0:
        raise ValueError("invalid 20-disk dimensions or spacing")
    return _compute_left_ventricle_volume_by_instant(a,b,float(step_size))

def validate_lv_mask(mask, spacing):
    if mask is None:
        raise ValueError("missing LV mask")
    a=np.asarray(mask)
    if a.ndim!=2 or min(a.shape)<2 or not np.isfinite(a).all() or not np.isin(a,[0,1]).all() or not a.any():
        raise ValueError("LV mask must be a nonempty finite 2D binary array")
    s=np.asarray(spacing,dtype=float)
    if s.shape!=(2,) or not np.isfinite(s).all() or (s<=0).any():
        raise ValueError("spacing must be two positive finite mm values in H,W order")
    _,n=ndimage.label(a,structure=np.ones((3,3),dtype=np.uint8))
    if n!=1:
        raise ValueError(f"LV mask has {n} components; no component removal is permitted")
    return np.array(a,dtype=np.uint8,copy=True),tuple(map(float,s))

def biplane_report(lv_masks: Mapping[str, Any], spacing_by_key: Mapping[str, Sequence[float]]) -> dict:
    """Four native binary masks and four native H,W mm spacings; failures are null."""
    report={"method":"CAMUS official 20-disk biplane Simpson",
            "citation":CITATION,"license":LICENSE,"edv_ml":None,"esv_ml":None,
            "ef_percent":None,"reason":None,"fallback_used":False}
    try:
        clean={}; spaces={}
        for key in ("2CH_ED","2CH_ES","4CH_ED","4CH_ES"):
            try:clean[key],spaces[key]=validate_lv_mask(lv_masks.get(key),spacing_by_key.get(key))
            except Exception as e:raise ValueError(f"{key}: {e}") from e
        for view in ("2CH","4CH"):
            if clean[f"{view}_ED"].shape!=clean[f"{view}_ES"].shape or not np.allclose(spaces[f"{view}_ED"],spaces[f"{view}_ES"],rtol=0,atol=1e-5):
                raise ValueError(f"{view}: ED/ES native geometry differs")
        edv,esv=compute_left_ventricle_volumes(clean["2CH_ED"],clean["2CH_ES"],spaces["2CH_ED"],
                                            clean["4CH_ED"],clean["4CH_ES"],spaces["4CH_ED"])
        if not math.isfinite(edv) or not math.isfinite(esv) or edv<=0 or esv<0:
            raise ValueError("invalid output volume")
        report.update(edv_ml=float(edv),esv_ml=float(esv))
        if esv>edv:report["reason"]="ESV exceeds EDV; EF is undefined for this geometry"
        else:report["ef_percent"]=100.0*(edv-esv)/edv
    except Exception as e:
        report["reason"]=f"{type(e).__name__}: {e}"
    return report
