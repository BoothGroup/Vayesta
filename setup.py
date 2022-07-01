#!/usr/bin/env python3

import os
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.test import test
from setuptools.command.build_ext import build_ext

setup_src = os.path.abspath(os.path.join(__file__, ".."))


class CMakeBuild(build_ext):
    def build_cmake(self, ext):
        src = os.path.join(setup_src, "vayesta", "libs")

        cmake_args = [f"-S{src}", f"-B{self.build_temp}"]
        build_args = []

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        if getattr(self, "parallel", False):
            build_args.append(f"-j{self.parallel}")

        subprocess.check_call(
                ["cmake", ext.sourcedir, *cmake_args],
                cwd=self.build_temp,
        )
        subprocess.check_call(
                ["cmake", "--build", ".", *build_args],
                cwd=self.build_temp,
        )


class DiscoverTests(test):
    user_options = [
            ("include-veryslow", "i", "Include tests marked as veryslow"),
    ]

    def initialize_options(self):
        test.initialize_options(self)
        self.include_veryslow = False

    def finalize_options(self):
        pass

    def run_tests(self):
        # Only import pytest in this scope
        import pytest

        src = os.path.join(setup_src, "vayesta", "tests")

        test_args = []
        if not self.include_veryslow:
            test_args.append("-m not veryslow")

        pytest.main([src, *test_args])



setup(
    name="Vayesta",
    version="0.0.0",
    description="A toolkit for quantum embedding methods",
    url="https://vayesta.github.io",
    download_url="https://github.com/BoothGroup/Vayesta",
    keywords=[
            "embedding",
            "quantum", "chemistry",
            "material", "science",
            "electronic", "structure",
            "dmet", "rpa",
    ],
    author=", ".join([
            "M. Nusspickel",
            "O. J. Backhouse",
            "B. Ibrahim",
            "A. Santana-Bonilla",
            "C. J. C. Scott",
            "G. H. Booth",
    ]),
    author_email="vayesta.embedding@gmail.com",
    license="Apache License 2.0",  # FIXME?
    platforms=[
            "Linux",
            "Mac OS-X",
    ],
    python_requires=">=3.7",
    classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "License :: OSI Approved :: Apache Software License",  # FIXME?
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: C",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
    ],
    package=find_packages(exclude=["*test*", "*examples*"]),
    include_package_data=True,
    install_requires=[
            "numpy>=1.19.0",
            "scipy>=1.2",
            "h5py>=2.7",
            "cvxpy>=1.1",
            "mpi4py>=2.0.0",
            "pyscf @ git+https://github.com/pyscf/pyscf@master#egg=pyscf",  # FIXME when pyscf wheels update
    ],
    ext_modules=[Extension("vayesta_lib", [])],
    cmdclass={
            "build_ext": CMakeBuild,
            "test": DiscoverTests,
    },
    tests_require=[
            "pytest",
            "pytest-cov",
    ],
    zip_safe=False,
)
