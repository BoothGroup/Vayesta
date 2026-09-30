import pytest
import os
import sys

src = os.path.abspath(os.path.join(__file__, "..", "..", "..", "vayesta"))

if __name__ == "__main__":
    # Run tests in parallel, with two OpenMP threads per worker, such that the OpenMP code is still tested
    # with multiple threads without oversubscribing the CPUs
    nthreads = int(os.environ.setdefault("OMP_NUM_THREADS", "2"))
    nworkers = max((os.cpu_count() or 1) // nthreads, 1)
    args = [
        "vayesta/tests",
        "--cov=vayesta",
        f"--numprocesses={nworkers}",
    ]

    if len(sys.argv) > 1 and sys.argv[1] == "--with-veryslow":
        args.append("-m veryslow or not veryslow")

    raise SystemExit(pytest.main(args))
