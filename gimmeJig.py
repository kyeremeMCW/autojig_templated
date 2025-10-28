from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Header, Nifti1Image
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
        zooms = self.header.get_zooms()
        return tuple(float(v) for v in zooms[: self.ndim])

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
