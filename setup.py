#!/usr/bin/env python3

import os
import glob
import shlex
import shutil
import subprocess
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.test import test
from setuptools.command.build_ext import build_ext

setup_src = os.path.abspath(os.path.join(__file__, ".."))

# TODO: mpi4py as optional extra


class CMakeExtension(Extension):
    """Initialise the name of a CMake extension.
    """

    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    """Build and configure a CMake extension.
    """

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        src = os.path.join(setup_src, "vayesta", "libs")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_args = [f"-S{src}", f"-B{self.build_temp}"]
        if os.getenv("CMAKE_CONFIGURE_ARGS"):
            cmake_args += os.getenv("CMAKE_CONFIGURE_ARGS").split()

        self.announce("Configuring")
        self.spawn(["cmake", *cmake_args])

        build_args = []
        if os.getenv("CMAKE_BUILD_ARGS"):
            cmake_args += os.getenv("CMAKE_BUILD_ARGS").split()
        if getattr(self, "parallel", False):
            build_args.append(f"-j{self.parallel}")

        self.announce("Building")
        self.spawn(["cmake", "--build", self.build_temp, *build_args])

    def get_ext_filename(self, ext_name):
        ext_path = os.path.join(*ext_name.split("."))
        fname = build_ext.get_ext_filename(self, ext_name)
        suffix = os.path.splitext(fname)[1]
        return ext_path + suffix


class CleanCommand(Command):
    """Clean up files resulting from compilation except for .so shared objects.
    """

    CLEAN_FILES = ["build", "dist", "*.egg-info"]
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.CLEAN_FILES:
            paths = glob.glob(os.path.normpath(os.path.join(setup_src, path_spec)))
            for path in paths:
                if not str(path).startswith(setup_src):
                    # In case CLEAN_FILES contains an address outside the package
                    raise ValueError("%s is not a path inside %s" % (path, setup_src))
                shutil.rmtree(path)


class DiscoverTests(test):
    """Discover and dispatch tests.
    """

    user_options = [
            ("include-veryslow", "v", "Include tests marked as veryslow"),
            ("include-slow", "s", "Include tests marked as slow"),
            ("pytest-args=", "p", "Extra arguments for pytest"),
    ]

    def initialize_options(self):
        test.initialize_options(self)
        self.include_veryslow = False
        self.include_slow = True
        self.pytest_args = ""

    def finalize_options(self):
        pass

    def run_tests(self):
        # Only import pytest in this scope
        import pytest

        src = os.path.join(setup_src, "vayesta", "tests")

        test_args = []
        if not (self.include_slow and self.include_veryslow):
            test_args.append("-m not (slow or veryslow)")
        elif not self.include_veryslow:
            test_args.append("-m not veryslow")
        elif not self.include_slow:
            test_args.append("-m not slow")
        test_args += shlex.split(self.pytest_args)

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
    packages=find_packages(exclude=["*tests*", "*examples*"]),
    include_package_data=True,
    install_requires=[
            "wheel",
            "numpy>=1.19.0",
            "scipy==1.1.0",  # pyscf needs >=1.1.0, cvxpy needs <=1.1.0
            "h5py>=2.7",
            "cvxpy>=1.1",
            "pyscf @ git+https://github.com/BoothGroup/pyscf@master",
            #"pyscf==2.0.1",
    ],
    ext_modules=[CMakeExtension("vayesta/libs")],
    cmdclass={
            "build_ext": CMakeBuild,
            "test": DiscoverTests,
            "clean": CleanCommand,
    },
    tests_require=[
            "pytest",
            "pytest-cov",
    ],
    zip_safe=False,
)
