from devito import Grid, Function, TimeFunction
from devito.data.allocators import ExternalAllocator
import numpy as np

def generate_data(shape, space_order=0, save=1):
    """
    When creating a Function/TimeFunction with a ExternalAllocator, 
    the shape of the initialized grid must consider the length 
    of the borders imposed by space_order and the shape of the array 
    given as input. 
    
    Paramdeters
    ----------
    array : numpy.array
        Array of data.
    space_order : int
        Space order of a Function/TimeFunction.

    Returns
    -------
    Grid
        Grid with specified shape. 

    Grid 

    Notes
    -----
    Tested for 2D cases.
    """
    grid_shape = tuple(x-2*y for x,y in zip(shape, (space_order,)*2))
    numpy_array = np.zeros(shape=(save,) + shape, dtype=np.float32)

    return (numpy_array, Grid(grid_shape))

def external_initializer(x):
    return None