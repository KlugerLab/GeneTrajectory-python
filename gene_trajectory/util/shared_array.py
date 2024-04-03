from multiprocessing.shared_memory import SharedMemory
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from numpy._typing import DTypeLike, _ShapeLike  # noqa
from typing import NamedTuple, Iterable, Callable, Any


class SharedArray(NamedTuple):
    """
    A pointer to a numpy.ndarray in shared memory
    For details, see https://docs.python.org/3/library/multiprocessing.shared_memory.html

    Example that sums a matrix by summing each row in parallel:
    - create a SharedMemoryManager, otherwise the shared memory would not be freed
    - create a shared copy of the array SharedArray.copy(manager, x)
    - pool workers unpack the shared array with the as_array() method
    - use pool.imap to get a generator for tdqm

    ```
    def sum_row(sai):
      sa, i = sai
      x = sa.as_array()
      return x[i].sum()

    def sum_by_row():
        x = np.ones([1000, 1000])
        with SharedMemoryManager() as manager:
            snp = SharedArray.copy(manager, x)
            with Pool() as pool:
                result_generator = pool.imap(sum_row, ((snp, i) for i in range(1000)))
                result_generator = tqdm(result_generator, total=1000, position=0, leave=True)
                return sum(result_generator)
    ```
    """
    shm: SharedMemory
    shape: _ShapeLike
    dtype: DTypeLike

    def as_array(self) -> np.ndarray:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    @staticmethod
    def allocate_like(manager: SharedMemoryManager, array: np.ndarray):
        shm = manager.SharedMemory(array.nbytes)
        return SharedArray(shm=shm, shape=array.shape, dtype=array.dtype)

    @staticmethod
    def allocate(manager: SharedMemoryManager, shape: _ShapeLike, dtype: DTypeLike, nbytes=None):
        nbytes = nbytes or np.zeros(1, dtype=dtype).nbytes
        shm = manager.SharedMemory(np.prod(shape) * nbytes) # noqa
        return SharedArray(shm=shm, shape=shape, dtype=dtype)

    @staticmethod
    def copy(manager: SharedMemoryManager, array: np.ndarray):
        sa = SharedArray.allocate_like(manager, array)
        a = sa.as_array()
        a[:] = array[:]
        return sa


class PartialStarApply:
    """
    wraps a function call to a function of np.array f(x1, x2, ..., arg1, arg2, ...)
    where the arrays x1, x2, ... are SharedArray and args are passed as a tuple (arg1, arg2, ...)

    Intended for use with multithreading.Pool.imap
    """

    __slots__ = "func", "args", "__dict__", "__weakref__"
    func: Callable[[Iterable], Any]
    args: Iterable[SharedArray]

    def unpacked_args(self) -> list[SharedArray]:
        return [a.as_array() if isinstance(a, SharedArray) else a for a in self.args]

    def __init__(self, func: Callable[..., Any], *args: SharedArray):
        assert isinstance(func, Callable)
        for a in args:
            assert isinstance(a, SharedArray)
        self.func = func
        self.args = args

    def __call__(self, args: Iterable):
        return self.func(*self.unpacked_args(), *args)