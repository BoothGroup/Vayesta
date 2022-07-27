import pytest
import sys
import os

src = os.path.abspath(os.path.join(__file__, "..", "..", "..", "vayesta"))

if __name__ == "__main__":
    args = [
        "vayesta/tests",
        "--cov=vayesta",
    ]

    if len(sys.argv) == 1 or sys.argv[1] != "--with-veryslow":
        args.append("-m not veryslow")

    pytest.main(args)
