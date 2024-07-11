"""_summary_"""
import time
import logging
from typing import cast
import argparse
import tempfile

import nibabel as nib
import numpy as np
import vtk
import cadquery as cq

from stl.mesh import Mesh
# pylint: disable=no-member
JIG_TRANSLATEY = 15
SLICINGZ = 1.2  # constant for now, args/custom later
X_WALL = 5
Y_WALL = 3
Z_WALL = 3

DEFAULT_NIFTI_PATH = (
    "/Volumes/Siren/Prostate_data/1556/MRI/Processed/prostate_mask.nii.gz"
)
DEFAULT_MOLD_STL_PATH = (
    "/Volumes/Siren/Prostate_data/1556/MRI/Processed/autojigger_mold.stl"
)
DEFAULT_JIG_STL_PATH = (
    "/Volumes/Siren/Prostate_data/1556/MRI/Processed/autojigger_slicer.stl"
)


# todo: pyproject.toml WIP
# todo: pypi upload/ github actions WIP
# todo: docs WIP
# todo: profiles (eventually) WIP


def find_slice_thickness(nifti_path: str) -> np.float32:
    """Finds the slice thickness of the given nifti file

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    np.float32
        Slice thickness for the given nifti as a float
    """
    loaded_nifti = nib.load(nifti_path)
    slice_thickness = np.round(loaded_nifti.header["pixdim"][3], 1)

    return slice_thickness


def create_reader(nifti_path: str) -> vtk.vtkNIFTIImageReader:
    """Creates a vtk object that reads nifti images

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    vtk.vtkNIFTIImageReader
        Reader object as a vtkNIFTIImageReader
    """
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_path)
    reader.Update()

    return reader


def create_surface_extractor(
    connect_port: vtk.vtkAlgorithmOutput, label: int = 1
) -> vtk.vtkDiscreteMarchingCubes:
    """Creates a vtk object that extracts the surface mesh from a nifti

    Parameters
    ----------
    connect_port : vtk.vtkAlgorithmOutput
        Output port from vtk object
    label : int, optional
        Label value specifying which surface to extract from the input. 
        Defaults to 1

    Returns
    -------
    vtk.vtkDiscreteMarchingCubes
        Surface extractor object as vtkDiscreteMarchingCubes. Extracts surface with specified label

    """
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(connect_port)
    surf.SetValue(0, int(label))
    surf.Update()


    return surf


def create_smoother(
    connect_port: vtk.vtkAlgorithmOutput, iterations: int = 10000
) -> vtk.vtkSmoothPolyDataFilter:
    """Creates a vtk object that smooths a surface mesh

    Parameters
    ----------
    connect_port : vtk.vtkAlgorithmOutput
        Output port from vtk object
    iterations : int, optional
        Number of iterations for smoothing.
        Defaults to 10000

    Returns
    -------
    vtk.vtkSmoothPolyDataFilter
        Smoother object as vtkSmoothPolyDataFilter. Smoothes a surface mesh per iteration
    """
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(connect_port)
    smoother.SetNumberOfIterations(iterations)
    smoother.Update()

    return smoother


def create_decimator(
    connect_port: vtk.vtkAlgorithmOutput, target_reduction: float = 0.98
) -> vtk.vtkDecimatePro:
    """Creates a vtk object that decimates a surface mesh, reducing the amount of polygons.

    Parameters
    ----------
    connect_port : vtk.vtkAlgorithmOutput
        Output port from vtk object.
    target_reduction : float, optional
        Target reduction of polygons in the mesh.
        A value between 0 and 1, where 0 means no reduction and 1 means complete reduction.
        Defaults to .98

    Returns
    -------
    vtk.vtkDecimatePro
        Decimator object as vtkDecimatePro. Decimates a surface mesh by target reduction.
    """
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(connect_port)
    decimate.SetTargetReduction(target_reduction)
    decimate.Update()

    return decimate


def create_scaler(
    connect_port: vtk.vtkAlgorithmOutput, scaling_factors: tuple = (1.02, 1.02, 1.02)
) -> vtk.vtkTransformPolyDataFilter:
    """Creates a vtk object that scales a surface mesh

    Parameters
    ----------
    connect_port : vtk.vtkAlgorithmOutput
        Output port from vtk object
    scaling_factors : tuple, optional
        3 index tuple containing the scaling factors for the (x, y, z) dimensions.
        Defaults to (1.02, 1.02, 1.02).

    Returns
    -------
    vtk.vtkTransformPolyDataFilter
        Scaler object as a vtkTransformPolyDataFilter. 
        Scales the surface mesh by the scaling factors
    """
    set_scaler = vtk.vtkTransform()
    set_scaler.Scale(scaling_factors)

    scaler = vtk.vtkTransformPolyDataFilter()
    scaler.SetTransform(set_scaler)
    scaler.SetInputConnection(connect_port)
    scaler.Update()

    return scaler


def find_bounds(poly_data: vtk.vtkPolyData) -> tuple:
    """Finds the boundaries of given vtk polydata

    Parameters
    ----------
    poly_data : vtk.vtkPolyData
        Input vtkPolyData object

    Returns
    -------
    tuple
       6 index tuple containing the boundaries of the polydata.
        (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    final_obj_bounds = poly_data.GetBounds()

    return final_obj_bounds


def find_translation(final_obj_bounds: tuple) -> tuple:
    """Calculates the translation needed to center the polydata based on its boundaries

    Parameters
    ----------
    final_obj_bounds : tuple
        6 index tuple containing the boundaries of the polydata.
        (xmin, xmax, ymin, ymax, zmin, zmax)

    Returns
    -------
    tuple
        3 index tuple representing the translation vector to move the center of the object to 0,0,0
        (x, y, z) 

    """
    center_x = (final_obj_bounds[0] + final_obj_bounds[1]) / 2
    center_y = (final_obj_bounds[2] + final_obj_bounds[3]) / 2
    center_z = (final_obj_bounds[4] + final_obj_bounds[5]) / 2

    translation = (-center_x, -center_y, -center_z)

    return translation


def create_translator(
    connect_port: vtk.vtkAlgorithmOutput, translation: tuple
) -> vtk.vtkTransformPolyDataFilter:
    """Creates a vtk object to translate a surface mesh

    Parameters
    ----------
    connect_port : vtk.vtkAlgorithmOutput
        Output port from vtk object
    translation : tuple
        3 index tuple containing the translation vectors specifying how much to move the surface mesh along each axis
        (x, y, z) 

    Returns
    -------
    vtk.vtkTransformPolyDataFilter
        Translator object as a vtkTransformPolyDataFilter. Translates the surface mesh by given translation

    """
    translation_transform = vtk.vtkTransform()
    translation_transform.Translate(translation)

    translation_filter = vtk.vtkTransformPolyDataFilter()
    translation_filter.SetTransform(translation_transform)
    translation_filter.SetInputConnection(connect_port)
    translation_filter.Update()

    return translation_filter


def prep_mold(nifti_path: str) -> vtk.vtkPolyData:
    """Creates a vtkPolyData object from the given nifti

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    vtk.vtkPolyData
        Surface mesh of the mold as a vtkPolyData object
    """
    reader = create_reader(nifti_path)
    surf = create_surface_extractor(reader.GetOutputPort())
    smoother = create_smoother(surf.GetOutputPort())
    decimate = create_decimator(smoother.GetOutputPort())
    scaler = create_scaler(decimate.GetOutputPort())

    poly_data = scaler.GetOutput()

    final_obj_bounds = find_bounds(poly_data)

    translation = find_translation(final_obj_bounds)
    translation_filter = create_translator(scaler.GetOutputPort(), translation)

    poly_data = translation_filter.GetOutput()

    return poly_data


def write_stl(poly_data: vtk.vtkPolyData, mold_stl_path: str) -> str:
    """Writes a vtkPolyData object to an STL file

    Parameters
    ----------
    poly_data : vtk.vtkPolyData
        Input vtkPolyData object of the surface mesh to be saved
    mold_stl_path : str
        File path to where the STL file will be saved

    Returns
    -------
    str
        File path of the saved STL file
    """
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(poly_data)
    writer.SetFileTypeToASCII()
    writer.SetFileName(mold_stl_path)
    writer.Write()
    return mold_stl_path


def to_vectors(points: tuple) -> tuple:  # from cqmore
    """Converts the given tuple of points to a tuple of CADQuery vectors

    Parameters
    ----------
    points : tuple
        Tuple of points representing the points in the object


    Returns
    -------
    tuple
        Tuple of CADQuery vectors, with size determined by the amount of points
    """
    if isinstance(next(iter(points)), cq.Vector):
        return cast(tuple[cq.Vector], list(points))

    return cast(tuple[cq.Vector], tuple(cq.Vector(*p) for p in points))


def make_polyhedron(points: tuple, faces: list) -> cq.Solid:  # from cqmore
    """Creates a polyhedral solid from given points and faces

    Parameters
    ----------
    points : tuple
        Tuple of points representing the points in the object
    faces : list
        List of lists with each sublist containing the vectors of each face

    Returns
    -------
    cq.Solid
        CADQuery Solid object created with the provided points and faces.
    """
    vectors = np.array(to_vectors(points))

    return cq.Solid.makeSolid(
        cq.Shell.makeShell(
            cq.Face.makeFromWires(
                cq.Wire.assembleEdges(
                    cq.Edge.makeLine(*vts[[-1 + i, i]]) for i in range(vts.size)
                )
            )
            for vts in (vectors[list(face)] for face in faces)
        )
    )


def find_jig_bounds(nifti_path: str) -> tuple:
    """Calculates the bounds of a jig based on a NIfTI file.

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    tuple
        6 index tuple representing the boundaries of the jig
        (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    poly_data = prep_mold(nifti_path)
    final_obj_bounds = find_bounds(poly_data)
    jig_modifiers = find_jig_modifiers(nifti_path)

    jig_bounds = (
        final_obj_bounds[0] + jig_modifiers[0],
        final_obj_bounds[1] + jig_modifiers[1],
        final_obj_bounds[2] + jig_modifiers[2],
        final_obj_bounds[3] + jig_modifiers[3],
        final_obj_bounds[4] + jig_modifiers[4],
        final_obj_bounds[5] + jig_modifiers[5],
    )

    return jig_bounds


def find_jig_size(nifti_path: str) -> tuple:
    """Determines the size dimensions of a jig from the nifti

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    tuple
        3 index tuple representing the dimensions of the jig
        (x, y, z) 
    """
    jig_bounds = find_jig_bounds(nifti_path)

    jigsizex = jig_bounds[1] - jig_bounds[0]
    jigsizey = jig_bounds[3] - jig_bounds[2]
    jigsizez = jig_bounds[5] - jig_bounds[4]

    jig_size = (jigsizex, jigsizey, jigsizez)

    return jig_size


def find_jig_modifiers(nifti_path: str) -> tuple:
    """Determines modifiers for creating a jig based on parameters extracted from the nifti

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    tuple
        6 index tuple of modifiers used for adjusting jig dimensions
        (-x, +x, -y, +y, -z, +z) 
    """
    slice_thickness = find_slice_thickness(nifti_path)

    pre_knife_slot = 37
    post_knife_slot = 3

    jig_modifiers = (
        -X_WALL,
        X_WALL,
        -(Y_WALL + post_knife_slot),
        Y_WALL + pre_knife_slot,
        -((slice_thickness) + Z_WALL),
        ((slice_thickness) + Z_WALL),
    )

    return jig_modifiers


def process_jig(nifti_path: str) -> cq.Workplane:
    """Creates a CAD object jig using the given nifti to find the appropriate size

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    cq.Workplane
        Jig as a CADquery Workplane object

    """

    jig_size = find_jig_size(nifti_path)

    result = cq.Workplane("XY").box(*jig_size).translate((0, JIG_TRANSLATEY, 0))

    cad_jig = result
    return cad_jig


def process_slicer(nifti_path: str) -> cq.Workplane:
    """Creates a CAD object slicer to slice the jig

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti

    Returns
    -------
    cq.Workplane
        Slicer as a CADquery Workplane object
    """
    jig_size = find_jig_size(nifti_path)
    slicingx = jig_size[0] + 2 * X_WALL
    slicingy = jig_size[1] - 2 * Y_WALL

    result = (
        cq.Workplane("XY")
        .box(slicingx, slicingy, SLICINGZ)
        .translate((0, JIG_TRANSLATEY, -((jig_size[2] / 2) - 3)))
    )

    cad_slicer = result
    return cad_slicer

 # see https://github.com/JustinSDK/cqMore/blob/main/examples/import_stl.py
def import_stl(stl_path: str) -> cq.Solid: 
    """Imports an STL file as a mesh for CADquery

    Parameters
    ----------
    stl_path : str
        File path to the specified nifti

    Returns
    -------
    cq.Solid
        CADquery Solid object representing the geometry from the STL file

    """
    vectors = Mesh.from_file(
        stl_path, remove_duplicate_polygons=True, remove_empty_areas=True
    ).vectors
    points = tuple(
        map(tuple, vectors.reshape((vectors.shape[0] * vectors.shape[1], 3)))
    )
    faces = [(i, i + 1, i + 2) for i in range(0, len(points), 3)]
    return make_polyhedron(points, faces)


def process_mold(nifti_path: str, mold_stl_path: str) -> cq.Solid:
    """Creates a 3D model from a nifti and saves it to the specified output path

    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti
    mold_stl_path : str
        Output file path for saving the STL file of the processed mold.

    Returns
    -------
    cq.Solid
        3D model of the processed mold as a CadQuery Solid
    """
    poly_data = prep_mold(nifti_path)
    stl_path = write_stl(poly_data, mold_stl_path)
    stl_mold = import_stl(stl_path)
    return stl_mold


def assemble_jig(nifti_path: str, mold_stl_path: str) -> cq.Workplane:
    """
    Assembles a 3D model by combining a jig, mold, and slicer


    Parameters
    ----------
    nifti_path : str
        File path to the specified nifti as a string
    mold_stl_path : str
        File path for the STL file of the processed mold as a string

    Returns
    -------
    cq.Workplane
        Combined 3D model assmebly of the jig as a CadQuery Workplane
    """
    jig = process_jig(nifti_path)
    slicer = process_slicer(nifti_path)
    mold = process_mold(nifti_path, mold_stl_path)

    slice_thickness = find_slice_thickness(nifti_path)
    jig_size = find_jig_size(nifti_path)

    assembly = cq.Workplane("XY").add(jig)

    end_z = int(np.round(jig_size[2] - find_jig_modifiers(nifti_path)[5], 2) * 100)
    step_z = int(np.round(slice_thickness, 2) * 100)
    slice_time = 0
    for slots in range(0, end_z, step_z):
        slice_start = time.time()
        assembly = assembly.cut(slicer.translate((0, 0, slots / 100)))
        logging.error(time.time()-slice_start)
        slice_time += time.time()-slice_start

    print(f"Time to slicer jig: {slice_time} sec")

    end_y = int(np.round((jig_size[1]) - JIG_TRANSLATEY))

    mold = mold.scale(0.01)
    assembly = assembly.val().scale(0.01)
    mold_time = 0
    for i in range(0, end_y, 2):
        mold_start = time.time()
        assembly = assembly.cut(mold.translate((0, i / 100, 0)))
        logging.error(time.time()-mold_start)
        mold_time += time.time()-mold_start
        
    print(f"Time to cut mold: {mold_time} sec")
    assembly = assembly.scale(100)

    return assembly


def main(nifti: str, mold_stl_path: str, jig_stl_path: str) -> None:
    polydata = prep_mold(nifti)

    if mold_stl_path == "":
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=True) as tmp_file:
            mold_stl_path = tmp_file.name
            write_stl(polydata, mold_stl_path)
            resulting_assembly = assemble_jig(nifti, mold_stl_path)
    else:
        write_stl(polydata, mold_stl_path)
        resulting_assembly = assemble_jig(nifti, mold_stl_path)

    resulting_assembly.exportStl(jig_stl_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process NIfTI file and output STLs.")
    parser.add_argument(
        '-nii','--nifti',
        nargs="?",
        type=str,
        help="Path to the input NIfTI file",
    )
    parser.add_argument(
        '-mold','--mold_stl_path',
        nargs="?",
        type=str,
        default= "",
        help="Path to the output STL file (optional)",
    )
    parser.add_argument(
        '-jig','--jig_stl_path',
        nargs="?",
        type=str,
        help="Path to the output STL file",
    )

    args = parser.parse_args()

    main(args.nifti, args.mold_stl_path, args.jig_stl_path)
