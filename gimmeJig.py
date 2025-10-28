from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Header, Nifti1Image
from nibabel.affines import apply_affine, voxel_sizes
from nibabel.orientations import aff2axcodes
from scipy.ndimage import binary_fill_holes, generate_binary_structure, label

ArrayLike = np.ndarray
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass
class NiftiVolume:
    """Container bundling array data with its spatial metadata."""

    data: ArrayLike
    affine: np.ndarray
    header: Nifti1Header
    path: Path

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def voxel_spacing(self) -> Tuple[float, ...]:
        sizes = voxel_sizes(self.affine)
        return tuple(float(v) for v in sizes[: self.ndim])

    @property
    def orientation_codes(self) -> Tuple[str, ...]:
        # aff2axcodes is cheap and keeps downstream logic explicit.
        return aff2axcodes(self.affine)

    def summary(self) -> str:
        spacing = ", ".join(f"{v:.3f}" for v in self.voxel_spacing)
        orientation = "".join(self.orientation_codes)
        return (
            f"NiftiVolume(path={self.path}, shape={self.shape}, "
            f"voxel_spacing=({spacing}), orientation={orientation})"
        )


@dataclass
class MaskPreprocessConfig:
    """Configuration flags controlling mask cleanup."""

    binarize_threshold: Optional[float] = 0.5
    connectivity: int = 1
    fill_holes: bool = True
    take_largest_component: bool = True


@dataclass
class MaskPreprocessResult:
    """Outcome of mask preprocessing for downstream jig generation."""

    mask: BoolArray
    input_mask: BoolArray
    num_components: int
    kept_component_label: Optional[int]
    removed_voxels: int
    holes_filled_voxels: int
    original_voxel_count: int
    processed_voxel_count: int
    component_sizes: IntArray
    warnings: List[str]


@dataclass
class JigParameterConfig:
    """Parameters controlling jig sizing and orientation."""

    padding_mm: float | Sequence[float] = 5.0
    orientation_axis: Optional[int] = None
    use_physical_spacing: bool = True


@dataclass
class JigParameters:
    """Computed dimensions and transforms needed to build the jig shell."""

    mask_bbox_min_vox: np.ndarray
    mask_bbox_max_vox: np.ndarray
    jig_bbox_min_vox: np.ndarray
    jig_bbox_max_vox: np.ndarray
    mask_bbox_min_mm: np.ndarray
    mask_bbox_max_mm: np.ndarray
    jig_bbox_min_mm: np.ndarray
    jig_bbox_max_mm: np.ndarray
    mask_size_mm: np.ndarray
    jig_size_mm: np.ndarray
    padding_mm: np.ndarray
    orientation_axis: int
    voxel_spacing: np.ndarray
    orientation_codes: Tuple[str, str, str]
    orientation_reason: str


def load_nifti(
    path: str | Path,
    *,
    dtype: Optional[np.dtype] = None,
    ensure_c_contiguous: bool = True,
) -> NiftiVolume:
    """Load a NIfTI image from disk and return a structured volume."""

    nifti_path = Path(path).expanduser().resolve()
    if not nifti_path.exists():
        raise FileNotFoundError(f"No file found at {nifti_path}")

    image = cast(Nifti1Image, nib_load(str(nifti_path)))
    data = image.get_fdata(dtype=dtype) if dtype is not None else image.get_fdata()
    if ensure_c_contiguous:
        data = np.ascontiguousarray(data)

    affine = np.array(image.affine, copy=True)
    header = cast(Nifti1Header, image.header.copy())

    volume = NiftiVolume(
        data=data,
        affine=affine,
        header=header,
        path=nifti_path,
    )
    return volume


def preprocess_mask(
    volume: NiftiVolume,
    config: Optional[MaskPreprocessConfig] = None,
) -> MaskPreprocessResult:
    """Clean up a binary anatomy mask by CC filtering and hole filling."""

    cfg = config or MaskPreprocessConfig()

    if cfg.connectivity < 1:
        raise ValueError("Connectivity must be at least 1.")

    data = np.asarray(volume.data)
    squeezed = np.squeeze(data)
    if squeezed.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask after squeezing singleton dims, got shape {squeezed.shape}."
        )

    if squeezed.dtype == np.bool_ or squeezed.dtype == bool:
        base_mask = squeezed.astype(bool, copy=False)
    else:
        if cfg.binarize_threshold is None:
            base_mask = squeezed != 0
        else:
            base_mask = squeezed >= cfg.binarize_threshold
    base_mask = np.ascontiguousarray(base_mask, dtype=bool)

    original_voxel_count = int(base_mask.sum())
    if original_voxel_count == 0:
        raise ValueError(f"Mask from {volume.path} contains no foreground voxels.")

    if cfg.connectivity > base_mask.ndim:
        raise ValueError(
            f"Connectivity {cfg.connectivity} exceeds mask dimensionality {base_mask.ndim}."
        )

    structure = generate_binary_structure(base_mask.ndim, cfg.connectivity)
    labeled, num_components = label(base_mask, structure=structure)  # type: ignore[misc]
    labeled = cast(np.ndarray, labeled)
    num_components = int(num_components)
    component_sizes = np.bincount(labeled.ravel(), minlength=num_components + 1)[1:].astype(
        np.int64,
        copy=False,
    )

    processed_mask: BoolArray = base_mask.copy()
    kept_component_label: Optional[int] = None
    removed_voxels = 0
    messages: List[str] = []

    if cfg.take_largest_component:
        if num_components == 0 or component_sizes.size == 0:
            raise ValueError("Connected component analysis found no foreground components.")

        largest_idx = int(np.argmax(component_sizes))
        kept_component_label = largest_idx + 1

        if num_components > 1:
            processed_mask = labeled == kept_component_label
            removed_voxels = original_voxel_count - int(processed_mask.sum())
            if removed_voxels > 0:
                msg = (
                    f"{volume.path.name}: mask disconnected; kept component {kept_component_label} "
                    f"({int(component_sizes[largest_idx])} voxels) and removed {removed_voxels} voxel(s) "
                    "from other component(s)."
                )
                warnings.warn(msg)
                messages.append(msg)
        else:
            kept_component_label = 1

    holes_filled_voxels = 0
    if cfg.fill_holes:
        filled = binary_fill_holes(processed_mask)
        filled_bool = np.ascontiguousarray(filled, dtype=bool)
        holes_filled_voxels = int(filled_bool.sum() - processed_mask.sum())
        if holes_filled_voxels > 0:
            msg = f"{volume.path.name}: filled {holes_filled_voxels} internal voxel(s)."
            warnings.warn(msg)
            messages.append(msg)
        processed_mask = filled_bool

    processed_voxel_count = int(processed_mask.sum())

    return MaskPreprocessResult(
        mask=processed_mask,
        input_mask=base_mask.copy(),
        num_components=int(num_components),
        kept_component_label=kept_component_label,
        removed_voxels=removed_voxels,
        holes_filled_voxels=holes_filled_voxels,
        original_voxel_count=original_voxel_count,
        processed_voxel_count=processed_voxel_count,
        component_sizes=component_sizes.astype(np.int64, copy=True),
        warnings=messages,
    )


def _voxel_indices_from_mask(mask: BoolArray) -> np.ndarray:
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("Mask contains no true voxels for bounding box computation.")
    return indices


def _normalize_padding(padding: float | Sequence[float]) -> np.ndarray:
    if isinstance(padding, (int, float, np.floating)) and not isinstance(padding, bool):
        value = float(padding)
        arr = np.full(3, value, dtype=np.float64)
    else:
        sequence = cast(Sequence[float], padding)
        try:
            seq = tuple(sequence)
        except TypeError as err:
            raise ValueError("Padding must be a scalar or an iterable of three floats.") from err
        arr = np.asarray(seq, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(
                f"Padding sequence must contain exactly three values (x, y, z); got shape {arr.shape}."
            )
    if np.any(arr < 0):
        raise ValueError("Padding values must be non-negative.")
    return arr


def derive_orientation_axis(voxel_spacing: np.ndarray, override: Optional[int]) -> Tuple[int, str]:
    if override is not None:
        if override < 0 or override >= voxel_spacing.size:
            raise ValueError(
                f"Orientation override {override} out of range for spacing vector of length {voxel_spacing.size}."
            )
        return override, "user override"

    axis = int(np.argmax(voxel_spacing))
    return axis, "coarsest voxel dimension"


def compute_jig_parameters(
    volume: NiftiVolume,
    mask_result: MaskPreprocessResult,
    config: Optional[JigParameterConfig] = None,
) -> JigParameters:
    """Derive bounding boxes and orientation cues for jig construction."""

    cfg = config or JigParameterConfig()

    mask = mask_result.mask
    voxel_indices = _voxel_indices_from_mask(mask)
    min_indices = voxel_indices.min(axis=0)
    max_indices = voxel_indices.max(axis=0)

    voxel_spacing = np.array(volume.voxel_spacing, dtype=float)
    if voxel_spacing.size != 3:
        raise ValueError(f"Expected 3 spacing values, got {voxel_spacing.size}.")

    orientation_axis, orientation_reason = derive_orientation_axis(voxel_spacing, cfg.orientation_axis)

    mask_bbox_min_vox = min_indices.astype(np.float64)
    mask_bbox_max_vox = max_indices.astype(np.float64)

    # Physical-space bounds derived from voxel edges for robustness to affine rotations/translations.
    corner_grid = np.array(
        np.meshgrid(
            [mask_bbox_min_vox[0] - 0.5, mask_bbox_max_vox[0] + 0.5],
            [mask_bbox_min_vox[1] - 0.5, mask_bbox_max_vox[1] + 0.5],
            [mask_bbox_min_vox[2] - 0.5, mask_bbox_max_vox[2] + 0.5],
            indexing="ij",
        )
    )
    corner_points = corner_grid.reshape(3, -1).T
    corner_points_mm = apply_affine(volume.affine, corner_points)
    mask_bbox_min_mm = corner_points_mm.min(axis=0)
    mask_bbox_max_mm = corner_points_mm.max(axis=0)

    voxel_extent = mask_bbox_max_vox - mask_bbox_min_vox + 1.0
    mask_size_mm = voxel_extent * voxel_spacing

    padding_vec = _normalize_padding(cfg.padding_mm)
    half_padding = padding_vec / 2.0

    padding_vox = padding_vec / voxel_spacing
    half_padding_vox = padding_vox / 2.0

    jig_bbox_min_vox = mask_bbox_min_vox - half_padding_vox
    jig_bbox_max_vox = mask_bbox_max_vox + half_padding_vox

    jig_bbox_min_mm = mask_bbox_min_mm - half_padding
    jig_bbox_max_mm = mask_bbox_max_mm + half_padding
    jig_size_mm = jig_bbox_max_mm - jig_bbox_min_mm

    orientation_codes = tuple(volume.orientation_codes)
    if len(orientation_codes) != 3:
        raise ValueError(
            f"Expected 3 orientation codes for a 3D volume, got {orientation_codes}."
        )

    return JigParameters(
        mask_bbox_min_vox=mask_bbox_min_vox,
        mask_bbox_max_vox=mask_bbox_max_vox,
    jig_bbox_min_vox=jig_bbox_min_vox,
    jig_bbox_max_vox=jig_bbox_max_vox,
        mask_bbox_min_mm=mask_bbox_min_mm,
        mask_bbox_max_mm=mask_bbox_max_mm,
        jig_bbox_min_mm=jig_bbox_min_mm,
        jig_bbox_max_mm=jig_bbox_max_mm,
        mask_size_mm=mask_size_mm,
        jig_size_mm=jig_size_mm,
        padding_mm=padding_vec,
        orientation_axis=orientation_axis,
        voxel_spacing=voxel_spacing,
        orientation_codes=cast(Tuple[str, str, str], orientation_codes),
        orientation_reason=orientation_reason,
    )
