import whitematteranalysis as wma
import numpy as np

def readVtk(tractography_path):
    pd_tractography = wma.io.read_polydata(tractography_path)
    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(pd_tractography, points_per_fiber=15)
    # fiber_array_r, fiber_array_a, fiber_array_s have the same size: [number of fibers, points of each fiber]
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    return feat
