* Build an MRI slicing jig automatically from niftis instead of manually in Blender.
* The jig is a rectangular prism sized from the mask bounding box + padding.
* The jig must contain slots sized to the knife width, spaced so knife_width + spacing == mri_slice_thickness.
* The anatomical model must be manipulated so it can slide into the box: slightly larger than anatomy and with the opening sized to the thickest part until it rests.
* Allow configurations on spatial manipulation of the anatomical model
* Configurable profiles should control padding, knife thickness, slicer orientation, tolerance of anatomy sizing, etc.
* Slicer orientation can usually be inferred from the coarsest voxel dimension (pixdim); allow overrides.
* Slice thickness should be derived from the orientation
* Provide quick errors in obvious failure cases, ie: slice thickness and knife thickness can't be the same, otherwise there would be no guides
* if a mask is disconnected, operate on largest CC, printing a warning about how that was necessary
* fill holes in the volume, warning that there was a hole to be filled
* Output a watertight, manifold STL for the slicing jig
* Write the code in such a way where we can visualize each step in a jupyter notebook

* The tools decided on with their reasons:
    * nibabel: load NIfTI/header metadata; great docs and examples; I’m very comfortable here.
    * numpy + scipy.ndimage: fast array math, morphological ops, CC labeling; both are well‑documented and my go-tos for preprocessing.
    * scikit-image: regionprops, hole filling, marching cubes; API docs and gallery are excellent, and I use it often.
    * trimesh (plus meshio if needed): cleaning meshes, repairing watertightness, STL export; docs are pragmatic and I’m comfortable navigating them.
    * pyvista (VTK wrapper): convenient 3D plotting for notebook step visualization; solid gallery/examples and I’m comfortable with the basics.
    * matplotlib or plotly: quick 2D slice/orthogonal views in notebooks; both well-known, with matplotlib as my default.