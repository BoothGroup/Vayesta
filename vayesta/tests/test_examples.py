import sys
from pathlib import Path
import importlib

import pytest


example_path = Path(__file__).parent.parent.parent / 'examples'
examples_files = list(example_path.glob('**/*.py'))


@pytest.fixture(params=examples_files, ids=lambda x: x.name)
def example_file(request):
    return request.param


def test_example(example_file):
    spec = importlib.util.spec_from_file_location(example_file.name, str(example_file))
    example = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = example
    spec.loader.exec_module(example)
