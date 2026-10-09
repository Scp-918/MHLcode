from pathlib import Path
import sys

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    'udeco_uh_roi_edge_detection.ipynb',
    'udeco_interval_statistics.ipynb',
    'figures/udeco_publication_figures.ipynb',
    'figures/udeco_flowchart_assets.ipynb',
    'figures/udeco_watch_skin_physics.ipynb',
)


@pytest.mark.parametrize('relative', NOTEBOOKS)
@pytest.mark.parametrize('working_directory', ('', 'figures'))
def test_notebook_imports_from_root_or_figures(relative, working_directory, monkeypatch):
    monkeypatch.chdir(ROOT / working_directory)
    monkeypatch.setattr(sys, 'path', [p for p in sys.path if p not in ('', str(ROOT))])
    notebook = nbformat.read(ROOT / relative, as_version=4)
    first = next(c for c in notebook.cells if c.cell_type == 'code')
    namespace = {}
    exec(compile(first.source, relative, 'exec'), namespace)
    assert namespace['PROJECT_ROOT'].resolve() == ROOT
    assert str(ROOT) in sys.path


@pytest.mark.parametrize('relative', NOTEBOOKS)
def test_notebook_delivery_has_no_execution_outputs(relative):
    notebook = nbformat.read(ROOT / relative, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None
            assert not cell.outputs
