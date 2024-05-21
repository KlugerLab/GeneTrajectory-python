from typing import Optional

import numpy as np


def validate_not_none(obj, obj_name: str = 'input'):
    if obj is None:
        raise ValueError(f"{obj_name} is None")


def validate_matrix(
        m: np.array,
        obj_name: str = 'input',
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        shape: Optional[tuple[int, int]] = None,
        square: Optional[bool] = None,
        min_size: Optional[int] = 1,
        min_value: Optional = None,
        max_value: Optional = None,
):
    """
    Validates an input matrix
    @param m: the input matrix
    @param obj_name: the name of the object, for error reporting
    @param min_size: Minimum matrix size in each dimension. Defaults to 1, and will raise for an empty matrix
    @param nrows: Number of rows in the matrix
    @param ncols: Number of rows in the matrix
    @param shape: the expected shape of the matrix
    @param square: If True, ensures the matrix is square. If False, ensures the matrix is not
    @param min_value: Minimum value for each element
    @param max_value: Maximum value for each element
    """
    validate_not_none(m, obj_name=obj_name)

    if len(m.shape) != 2:
        raise ValueError(f"{obj_name} is not a matrix. Shape: {m.shape}")
    mr, mc = m.shape

    if nrows is not None:
        if mr != nrows:
            raise ValueError(f"{obj_name} does not have {nrows} rows. Shape: {m.shape}")

    if ncols is not None:
        if mc != ncols:
            raise ValueError(f"{obj_name} does not have {ncols} columns. Shape: {m.shape}")

    if shape is not None:
        if m.shape != shape:
            raise ValueError(f"{obj_name} does not have shape {shape}. Shape: {m.shape}")

    if square is True:
        if mr != mc:
            raise ValueError(f"{obj_name} is not a square matrix. Shape: {m.shape}")
    elif square is False:
        if mr == mc:
            raise ValueError(f"{obj_name} is a square matrix. Shape: {m.shape}")

    if min_size is not None:
        for s in m.shape:
            if s < min_size:
                raise ValueError(f"{obj_name} does not have enough elements. Min_size: {min_size}, Shape: {m.shape}")

    if min_value is not None:
        if m.min() < min_value:
            raise ValueError(f"{obj_name} should not have values less than {min_value}. Minimum found: {m.min()}")

    if max_value is not None:
        if m.max() > max_value:
            raise ValueError(f"{obj_name} should not have values greater than {max_value}. Maximum found: {m.max()}")
