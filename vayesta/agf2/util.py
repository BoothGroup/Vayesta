import numpy as np

def block_diagonal(arrays):
    ''' Constructs a block diagonal array from a series of arrays.
        Input arrays don't need to be square or the same shape.
    '''

    arrays = [np.asarray(array) for array in arrays]

    array = arrays[0]

    for i in range(1, len(arrays)):
        array_next = arrays[i]

        zeros_ur = np.zeros((array.shape[0], array_next.shape[1]), dtype=array.dtype)
        zeros_bl = np.zeros((array_next.shape[0], array.shape[1]), dtype=array.dtype)

        array = np.block([[array, zeros_ur], [zeros_bl, array_next]])

    return array



