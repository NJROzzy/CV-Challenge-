import hashlib
import json
import sys
from cvbench import benchmark


def test_worker_records_complete_measurements():
    config = dict(height=24, width=32, batch=4, seed=7, backend='cpu', kernel=5,
                  sigma=1.2, warmup=1, repetitions=3, threads=1)
    result = benchmark.worker(config)
    assert result['status'] == 'ok'
    assert result['input_sha256'] == hashlib.sha256(benchmark.workload(24, 32, 4, 7).tobytes()).hexdigest()
    times = result['end_to_end']['samples_ms']
    assert len(times) == 3 and min(times) > 0
    assert result['throughput_fps'] == 12 * 1000 / sum(times)
    assert result['memory']['process_peak_rss_bytes'] > 0
    assert result['memory']['cuda_peak_allocated_bytes'] is None
    assert all(len(s['samples_ms']) == 3 for s in result['stages'].values())


def test_unavailable_cuda_report_has_no_fabricated_results(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, 'environment', lambda: {'cuda_available': False})
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['cvbench', '--backend', 'gpu', '--resolutions', '480p',
                                    '--batches', '1', '--output', str(output)])
    benchmark.main()
    case = json.loads(output.read_text())['cases'][0]
    assert case['comparison'] is None
    gpu = case['backends']['gpu']
    assert gpu['status'] == 'unavailable'
    assert 'end_to_end' not in gpu and 'similarity' not in gpu
