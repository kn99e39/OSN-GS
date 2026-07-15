
import atexit
import importlib
import json
import os
import runpy
import sys
import time
import torch

SH_C0 = 0.28209479177387814
WS_URL = os.environ.get('GS_STREAM_WS_URL', '').strip()
STREAM_EVERY = int(os.environ.get('GS_STREAM_EVERY', '0') or '0')
STREAM_ITERATIONS = {int(value) for value in os.environ.get('GS_STREAM_ITERATIONS', '').split(',') if value.strip()}
STREAM_MAX_GAUSSIANS = int(os.environ.get('GS_STREAM_MAX_GAUSSIANS', '0') or '0')
STREAM_CACHE_DIR = os.environ.get('GS_STREAM_CACHE_DIR', '').strip()
TRAIN_SCRIPT = os.environ.get('GS_TRAIN_SCRIPT', 'train.py')
GAUSSIAN_MODEL_IMPORT = os.environ.get('GS_GAUSSIAN_MODEL_IMPORT', 'scene.gaussian_model:GaussianModel')
GAUSSIAN_STREAM_HOOK = os.environ.get('GS_GAUSSIAN_STREAM_HOOK', 'update_learning_rate')
_ws = None
_last_error_at = 0.0

def import_symbol(spec):
    module_name, _, attr_name = spec.partition(':')
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)

GaussianModel = import_symbol(GAUSSIAN_MODEL_IMPORT)

def should_stream(iteration):
    if not WS_URL and not STREAM_CACHE_DIR:
        return False
    if iteration in STREAM_ITERATIONS:
        return True
    return STREAM_EVERY > 0 and iteration % STREAM_EVERY == 0

def get_socket():
    global _ws
    if _ws is not None:
        return _ws
    from websockets.sync.client import connect
    _ws = connect(WS_URL, max_size=None, open_timeout=10, close_timeout=2)
    try:
        _ws.recv(timeout=1)
    except Exception:
        pass
    print(f'[WS] connected to trainer SSH tunnel: {WS_URL}', flush=True)
    return _ws

def close_socket():
    global _ws
    if _ws is not None:
        try:
            _ws.close()
        except Exception:
            pass
        _ws = None

atexit.register(close_socket)

def selected_indices(count, device):
    if STREAM_MAX_GAUSSIANS > 0 and count > STREAM_MAX_GAUSSIANS:
        return torch.linspace(0, count - 1, steps=STREAM_MAX_GAUSSIANS, device=device).long()
    return slice(None)

def snapshot_payload(gaussians, iteration):
    with torch.no_grad():
        xyz_all = gaussians.get_xyz
        total_count = int(xyz_all.shape[0])
        idx = selected_indices(total_count, xyz_all.device)
        xyz = xyz_all[idx].detach().float().cpu()
        scaling = gaussians.get_scaling[idx].detach().float().cpu()
        rotation = gaussians.get_rotation[idx].detach().float().cpu()
        opacity = gaussians.get_opacity[idx].detach().float().reshape(-1).cpu()
        fdc = gaussians.get_features_dc[idx, 0, :].detach().float().cpu()
        color = torch.clamp(0.5 + SH_C0 * fdc, 0.0, 1.0)
    count = int(xyz.shape[0])
    return {
        'type': 'snapshot',
        'iteration': int(iteration),
        'parameterSpace': 'render',
        'count': count,
        'positions': xyz.reshape(-1).tolist(),
        'scales': scaling.reshape(-1).tolist(),
        'colors': color.reshape(-1).tolist(),
        'opacities': opacity.reshape(-1).tolist(),
        'rotations': rotation.reshape(-1).tolist(),
        'metadata': {
            'source': 'colab-training',
            'totalCount': total_count,
            'sentCount': count,
            'capped': count != total_count,
        },
    }

def cache_snapshot(payload):
    if not STREAM_CACHE_DIR:
        return
    from pathlib import Path
    cache_dir = Path(STREAM_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    iteration = int(payload.get('iteration', 0))
    (cache_dir / f'{iteration:08d}.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

def stream_snapshot(gaussians, iteration):
    global _last_error_at
    if not should_stream(iteration):
        return
    try:
        payload = snapshot_payload(gaussians, iteration)
        cache_snapshot(payload)
        if not WS_URL:
            return
        get_socket().send(json.dumps(payload, separators=(',', ':')))
        capped = ' capped' if payload['metadata']['capped'] else ''
        print(f"[WS] sent iteration {iteration}: {payload['count']}/{payload['metadata']['totalCount']} gaussians{capped}", flush=True)
    except Exception as exc:
        now = time.time()
        if now - _last_error_at > 10:
            print(f'[WS] stream/cache failed at iteration {iteration}: {exc}', flush=True)
            _last_error_at = now
        close_socket()

_original_stream_hook = getattr(GaussianModel, GAUSSIAN_STREAM_HOOK)

def stream_hook_wrapper(self, iteration, *args, **kwargs):
    result = _original_stream_hook(self, iteration, *args, **kwargs)
    stream_snapshot(self, iteration)
    return result

setattr(GaussianModel, GAUSSIAN_STREAM_HOOK, stream_hook_wrapper)
sys.argv[0] = TRAIN_SCRIPT
runpy.run_path(TRAIN_SCRIPT, run_name='__main__')
