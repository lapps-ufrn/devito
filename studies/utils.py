from devito import Grid, Function, TimeFunction
from devito.data.allocators import ExternalAllocator
import numpy as np

def generate_data(shape, space_order=0, save=1):
    """
    Initialize a numpy array and a Grid object with compatible
    shapes, considering the length of the borders imposed by 
    space_order and the number of timesteps to be recorded.
    
    Parameters
    ----------
    shape: tuple of int
        Shape of the data grid, including implicit regions.
    space_order : int
        Space order of a Function/TimeFunction.
    save : int
        Number of stored timesteps of a TimeFunction.

    Returns
    -------
    numpy_array
        numpy_array with compatible shape.
    Grid
        Grid with specified shape. 

    Notes
    -----
    Tested for 2D cases only.
    """

    grid_shape = tuple(x-2*y for x,y in zip(shape, (space_order,)*2))
    numpy_array = np.zeros(shape=(save,) + shape, dtype=np.float32)

    return (numpy_array, Grid(grid_shape))

def external_initializer(x):
    return None