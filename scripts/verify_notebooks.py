"""Run the five retained notebooks in fresh PPG_sensor_env kernels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

NOTEBOOKS = (
    'udeco_uh_roi_edge_detection.ipynb',
    'udeco_interval_statistics.ipynb',
    'figures/udeco_flowchart_assets.ipynb',
    'figures/udeco_watch_skin_physics.ipynb',
    'figures/udeco_publication_figures.ipynb',
)

SUMMARY_CODE = '''
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import udeco_analysis

assert Path(sys.executable).resolve() == Path(EXPECTED_PYTHON).resolve(), sys.executable
assert Path(udeco_analysis.__file__).resolve().parent == Path(EXPECTED_ROOT).resolve()
summary = {'python': sys.executable, 'module_root': str(Path(udeco_analysis.__file__).resolve().parent)}
def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest()
if 'tables' in globals() and 'roi_results' in globals():
    summary['analysis'] = {
        'time': array_hash(time_s),
        'processed': {name: {key: array_hash(a) for key, a in result.items() if isinstance(a, np.ndarray)}
                      for name, result in processed.items()},
        'rois': {str(ch): json.loads(json.dumps(roi_results[ch]['rois'], default=lambda x: x.item())) for ch in (2, 3)},
        'tables': {str(ch): json.loads(tables[ch].to_json(orient='split', double_precision=15)) for ch in (2, 3)},
    }
    summary['roi_counts'] = {str(ch): len(roi_results[ch]['rois']) for ch in (2, 3)}
    summary['modes'] = {str(ch): tables[ch].groupby('roi_id').roi_mode.first().value_counts().to_dict() for ch in (2, 3)}
if 'validation_result' in globals():
    assert validation_result is not None, 'single-sample validation was skipped'
    assert RUN_BATCH is False and not batch_results
    summary['interval'] = {ch: validation_result['channels'][ch]['statistics'] for ch in ('Udeco2', 'Udeco3')}
if 'FIGURES' in globals():
    assert len(FIGURES) == 11, list(FIGURES)
    assert tuple(FIGURES['03_spectrum'].layout.xaxis.range) == (0., 10.)
    assert FIGURES['06_multimodal'].layout.height == 1240
    axes = {tr.xaxis for tr in FIGURES['08_baselines'].data}
    assert axes == {'x', 'x2', 'x3'}, axes
    assert not any('ACC' in str(tr.name) for tr in FIGURES['08_baselines'].data)
    for key, fig in FIGURES.items():
        for tr in fig.data:
            assert len(tr.x) == len(tr.y), (key, tr.name)
        for k in fig.layout:
            if k.startswith('xaxis') and k != 'xaxis' and key in ('06_multimodal', '07_separate_ppg', '08_baselines'):
                assert fig.layout[k].matches == 'x'
        labels = [str(tr.name) for tr in fig.data] + [str(a.text) for a in fig.layout.annotations]
        assert not any('cancel' in label.lower() for label in labels)
    freq, amp = amplitude_spectrum(1.7 * np.sin(2 * np.pi * 3 * np.arange(1000) / 100), 100)
    k = int(np.argmax(amp))
    assert freq[k] == 3 and abs(amp[k] - 1.7) < 1e-4
    for action in (lambda: resolve_input('__missing_sample__.csv'),
                   lambda: numeric_signal(pd.DataFrame(), 'PPG_G'),
                   lambda: plot_flowchart_cn('__missing_asset__.svg'),
                   lambda: plot_lowess_explanation('__missing_asset__.svg')):
        try:
            action()
        except (ValueError, FileNotFoundError, KeyError):
            pass
        else:
            raise AssertionError('missing input did not fail explicitly')
    original_rois = {ch: roi_results[ch]['rois'] for ch in (2, 3)}
    original_tables = tables.copy()
    for ch in (2, 3):
        roi_results[ch]['rois'] = []
        tables[ch] = tables[ch].iloc[:0].copy()
        plot_states(ch)
    plot_combined()
    plot_baselines()
    for ch in (2, 3):
        roi_results[ch]['rois'] = original_rois[ch]
        tables[ch] = original_tables[ch]
    PLOT_START_SEC, PLOT_END_SEC = 0., 1.
    plot_combined()
    summary['figure_checks'] = '11 Plotly figures, SVG 9/10, FFT, alignment, no-ROI and missing inputs passed'
if 'svg_path' in globals():
    assert Path(svg_path).is_file()
    summary['svg'] = str(svg_path)
Path(SUMMARY_PATH).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.item()), encoding='utf-8')
print('NOTEBOOK_VERIFIED', json.dumps({k: v for k, v in summary.items() if k not in ('analysis', 'interval')}, ensure_ascii=False))
'''


def execute_notebook(root: Path, relative: str, data_root: Path, sample: Path,
                     report_dir: Path, reference: Path | None, ksm: KernelSpecManager) -> dict:
    source_path = root / relative
    nb = nbformat.read(source_path, as_version=4)
    nbformat.validate(nb)
    summary_path = report_dir / (source_path.stem + '.summary.json')
    # These changes affect only the temporary execution copy, never user configuration.
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        src = cell.source
        overrides = []
        if source_path.stem == 'udeco_uh_roi_edge_detection' and 'BASE_CONFIG = ExperimentConfig(' in src:
            overrides = [f'DATA_ROOT = Path({str(data_root)!r})', f'SAMPLE_NAME = {sample.stem.removesuffix("_sensor")!r}']
        elif source_path.stem == 'udeco_interval_statistics' and 'RUN_BATCH =' in src:
            overrides = [f'DATA_ROOT = Path({str(data_root)!r})', f'VALIDATION_FILE = Path({str(sample)!r})', 'RUN_BATCH = False']
        elif source_path.stem == 'udeco_publication_figures' and 'SAMPLE_CSV =' in src:
            overrides = [f'SAMPLE_CSV = Path({str(sample)!r})', f'DATA_ROOT = Path({str(data_root)!r})']
        if overrides:
            cell.source += '\n\n# Verification-only input overrides\n' + '\n'.join(overrides)
        cell.outputs = []
        cell.execution_count = None
    bootstrap = f'''from pathlib import Path
import sys
EXPECTED_ROOT = {str(root)!r}
EXPECTED_PYTHON = {sys.executable!r}
SUMMARY_PATH = {str(summary_path)!r}
assert Path(sys.executable).resolve() == Path(EXPECTED_PYTHON).resolve()
# Notebook bootstrap must establish imports without help from the runner.
'''
    nb.cells.insert(0, nbformat.v4.new_code_cell(bootstrap))
    nb.cells.append(nbformat.v4.new_code_cell(SUMMARY_CODE))
    started = time.monotonic()
    print(f'RUN {relative}', flush=True)
    km = KernelManager(kernel_name='ppg_release', kernel_spec_manager=ksm)
    client = NotebookClient(nb, km=km, timeout=600, allow_errors=False,
                            resources={'metadata': {'path': str(source_path.parent)}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        if client.kc is not None:
            client.kc.stop_channels()
        km.cleanup_resources()
        nbformat.write(nb, report_dir / source_path.name)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if reference is not None:
        old_path = reference / summary_path.name
        if old_path.exists():
            old = json.loads(old_path.read_text(encoding='utf-8'))
            for key in ('analysis', 'interval'):
                if key in old:
                    assert summary[key] == old[key], f'{relative}: {key} differs from reference'
    print(f'PASS {relative} ({time.monotonic() - started:.1f}s)', flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--sample', default='cgx/cgx_rope_1_sensor.csv')
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--notebooks', nargs='+', default=list(NOTEBOOKS))
    args = parser.parse_args()
    root = args.project_root.resolve()
    data_root = (args.data_root or root / 'subject_data').resolve()
    sample = (data_root / args.sample).resolve()
    if not sample.is_file() or not (data_root / 'ck.mat').is_file():
        raise FileNotFoundError(f'sample or calibration missing: {sample}, {data_root / "ck.mat"}')
    if Path(sys.prefix).name.lower() != 'ppg_sensor_env':
        raise RuntimeError('Run this command using conda run -n PPG_sensor_env python')
    report_dir = (args.report_dir or root / 'tmp' / 'notebook-verification').resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir = report_dir / 'kernels' / 'ppg_release'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'PPG_sensor_env verification', 'language': 'python',
    }), encoding='utf-8')
    ksm = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    results = {}
    for relative in args.notebooks:
        results[relative] = execute_notebook(root, relative, data_root, sample,
                                             report_dir, args.reference_dir, ksm)
    (report_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'ALL PASSED: {len(results)} notebooks; reports: {report_dir}', flush=True)


if __name__ == '__main__':
    main()
