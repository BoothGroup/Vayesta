import pytest
import os
import sys

src = os.path.abspath(os.path.join(__file__, "..", "..", "..", "vayesta"))

if __name__ == "__main__":
    args = [
        "vayesta/tests",
        "--cov=vayesta",
        "-n auto",
    ]

    if len(sys.argv) > 1 and sys.argv[1] == "--with-veryslow":
        args.append("-m veryslow or not veryslow")

    raise SystemExit(pytest.main(args))
