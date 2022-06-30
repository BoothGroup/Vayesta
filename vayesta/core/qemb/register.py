from vayesta.mpi import mpi


class FragmentRegister:

    def __init__(self, mpi_size=None):
        self._next_id = -1
        if mpi_size is None:
            mpi_size = mpi.size
        self._mpi_size = mpi_size
        self._next_mpi_rank = -1

    def get_next_id(self):
        self._next_id += 1
        return self._next_id

    def get_next_mpi_rank(self, runtime=None, memory=None):
        """TODO: get next MPI rank based on runtime and memory estimates."""
        if runtime is not None:
            raise NotImplementedError()
        if memory is not None:
            raise NotImplementedError()
        self._next_mpi_rank = (self._next_mpi_rank + 1) % self._mpi_size
        return self._next_mpi_rank

    def get_next(self, *args, **kwargs):
        """Get next free fragment ID and MPI rank."""
        return (self.get_next_id(), self.get_next_mpi_rank(*args, **kwargs))
