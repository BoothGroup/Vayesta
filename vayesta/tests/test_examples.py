import sys
from pathlib import Path
import importlib
from time import perf_counter

import pytest


example_path = Path(__file__).parent.parent.parent / 'examples'
examples_files = list(example_path.glob('**/*.py'))
timings = {}


@pytest.fixture(params=examples_files, ids=lambda x: '/'.join([x.parent.name, x.name]))
def example_file(request):
    return request.param


@pytest.fixture(scope='module', autouse=True)
def report_timings() -> None:
    yield
    for name, time in timings.items():
        print(f"Time for {name:20s}: {time:.1f} s")


@pytest.mark.timeout(300)
def test_example(example_file):
    spec = importlib.util.spec_from_file_location(example_file.name, str(example_file))
    example = importlib.util.module_from_spec(spec)
    t_start = perf_counter()
    spec.loader.exec_module(example)
    timings['/'.join([example_file.parent.name, example_file.name])] = perf_counter() - t_start
