import os
import functools
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_cmake(self, ext):
        src = os.path.abspath(os.path.join(__file__, "..", "pyscf", "lib"))

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


setup(
    name="Vayesta",
    version="0.0.0",
    description="A toolkit for quantum embedding methods",
    url="https://vayesta.github.io",
    project_urls={
        # TODO tutorials, docs
        "source": "https://github.com/BoothGroup/Vayesta",
        "issues": "https://github.com/BoothGroup/Vayesta/issues",
    },
    keywords=[
        "embedding",
        "quantum", "chemistry",
        "material", "science",
        "electronic", "structure",
        "DMET", "RPA",
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
    classifier=[
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
    install_requires=[  # FIXME mpi4py, cvxpy
        "numpy>=1.19.0",
        "scipy>=1.2",
        "h5py>=2.7",
        "pyscf @ git+https://github.com/pyscf/pyscf@master#egg=pyscf",  # FIXME when pyscf wheels update
    ],
    tests_require=[
            "pytest",
            "pytest-cov",
    ],
    zip_safe=False,
)
