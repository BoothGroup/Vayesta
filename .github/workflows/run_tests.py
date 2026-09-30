import pytest
import os
import sys

src = os.path.abspath(os.path.join(__file__, "..", "..", "..", "vayesta"))

if __name__ == "__main__":
    # Run tests in parallel, with one BLAS/OpenMP thread per worker to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = [
        "vayesta/tests",
        "--cov=vayesta",
        "--numprocesses=auto",
    ]

    if len(sys.argv) > 1 and sys.argv[1] == "--with-veryslow":
        args.append("-m veryslow or not veryslow")

    raise SystemExit(pytest.main(args))
