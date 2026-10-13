"""frames2foam: drive OpenFOAM moving-boundary simulations from animation frames.

Reads a periodic sequence of surface frames (OBJ or STL) with consistent vertex
ordering, fits the motion in time (periodic cubic spline, trigonometric/Fourier,
or linear), checks the volume rate implied by the motion, and writes boundary
data for OpenFOAM's timeVaryingMappedFixedValue condition."""
from .io import read_frames, read_surface
from .fit import PeriodicMotion
from .geometry import surface_flux, signed_volume, SurfaceMap
from .foam import read_patch_points, write_boundary_data
__version__ = "0.1.0"
__all__ = ["read_frames", "read_surface", "PeriodicMotion", "surface_flux",
           "signed_volume", "SurfaceMap", "read_patch_points", "write_boundary_data"]
