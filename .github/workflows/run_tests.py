import pytest
import sys

if __name__ == "__main__":
    args = [
        "vayesta/tests",
        "--cov vayesta",
        "--cov-report xml",
        "--cov-report term",
        "--cov-config .coveragerc",

    if not sys.argv[1] == "--with-veryslow":
        args.append("-m not veryslow")

    pytest.main(args)
