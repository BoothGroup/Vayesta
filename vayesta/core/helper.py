from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike


def orbital_sign_convention(mo_coeff: np.ndarray, inplace: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if not inplace:
        mo_coeff = mo_coeff.copy()
    absmax = np.argmax(abs(mo_coeff), axis=0)
    nmo = mo_coeff.shape[-1]
    swap = mo_coeff[absmax, np.arange(nmo)] < 0
    mo_coeff[:, swap] *= -1
    signs = np.ones((nmo,), dtype=int)
    signs[swap] = -1
    return mo_coeff, signs


# --- Packing/unpacking arrays


def get_dtype_int(obj: np.ndarray | None) -> int:
    if obj is None:
        return 0
    dtint = np.asarray(obj.dtype.char, dtype="a8").view(int)[()]
    return int(dtint)


def get_dtype(dtint: int) -> np.dtype | None:
    if dtint == 0:
        return None
    val = np.asarray(dtint).view("a8")[()]
    dtype = np.dtype(val)
    return dtype


def pack_metadata(array: np.ndarray | None, maxdim: int = 8) -> np.ndarray:
    if np.ndim(array) > maxdim:
        raise NotImplementedError
    dtint = get_dtype_int(array)
    if array is not None:
        ndim = array.ndim
        shape = list(array.shape) + (maxdim - array.ndim) * [0]
    else:
        ndim = 0
        shape = maxdim * [0]
    metadata = [dtint, ndim] + shape
    return np.asarray(metadata, dtype=int)


def unpack_metadata(array: np.ndarray, maxdim: int = 8) -> tuple[np.dtype | None, int, np.ndarray]:
    metadata = array[: maxdim + 2].view(int)
    dtype = get_dtype(metadata[0])
    ndim, shape = metadata[1], metadata[2:]
    return dtype, ndim, shape


def pack_arrays(*arrays: np.ndarray | None, dtype: DTypeLike = float, maxdim: int = 8) -> np.ndarray:
    """Pack multiple arrays into a single array of data type `dtype`.

    Useful for MPI communication."""

    def pack(array: np.ndarray | None) -> np.ndarray:
        metadata = pack_metadata(array).view(dtype)
        if array is None:
            return metadata
        array = array.flatten().view(dtype)
        return np.hstack((metadata, array))

    packed = []
    for array in arrays:
        packed.append(pack(array))
    return np.hstack(packed)


def unpack_arrays(packed: np.ndarray, dtype: DTypeLike = float, maxdim: int = 8) -> list[np.ndarray | None]:
    """Unpack a single array of data type `dtype` into multiple arrays.

    Useful for MPI communication."""

    unpacked = []
    while True:
        if packed.size == 0:
            break
        metadata, packed = np.hsplit(packed, [maxdim + 2])
        dtype, ndim, shape = unpack_metadata(metadata)
        if dtype is None:
            unpacked.append(None)
            continue
        shape = shape[:ndim]
        size = np.prod(shape)
        array, packed = np.hsplit(packed, [size])
        unpacked.append(array.view(dtype).reshape(shape))
    return unpacked


if __name__ == "__main__":
    import sys

    arrays_in = [
        np.asarray(list(range(100))),
        None,
        np.random.rand(70),
        # np.random.rand(70)*1j,
        # np.asarray([True, False, False])
    ]
    pack = pack_arrays(*arrays_in)
    arrays_out = unpack_arrays(pack)
    assert len(arrays_in) == len(arrays_out)
    for i, x in enumerate(arrays_in):
        if x is None:
            assert arrays_out[i] is None
        else:
            assert np.all(x == arrays_out[i])

    1 / 0

    obj = np.random.rand(3)
    dtint = get_dtype_int(obj)
    dt = get_dtype(dtint)

    obj = np.asarray([2])
    dtint = get_dtype_int(obj)
    dt = get_dtype(dtint)

    obj = np.asarray([2], dtype=np.int8)
    dtint = get_dtype_int(obj)
    dt = get_dtype(dtint)

    obj = np.asarray([True])
    dtint = get_dtype_int(obj)
    dt = get_dtype(dtint)
