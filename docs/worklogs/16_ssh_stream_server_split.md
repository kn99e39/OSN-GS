# SSH Stream Server Split

## Work Performed

- Removed the trainer-owned WebSocket server path from `TorchOSNGSTrainer`.
- Kept training streaming as a WebSocket client path through `--stream_url`.
- Replaced `scripts/start_trainer_stream.ps1` with a standalone local stream server launcher.
- Reworked `osn_gs/interop/trainer_ws_server.py` into a loopback-only WebSocket stream server that accepts snapshot JSON from training clients and broadcasts it to renderer clients.
- Updated `WebRenderer/README.txt` with the SSH local port-forward workflow.

## Result

The streaming path is now separate from training again:

```text
OSN-GS train.py / notebook --stream_url ws://127.0.0.1:8080
  -> local stream server on trainer host
  -> SSH local port forwarding
  -> renderer browser ws://localhost:8080
```

The training loop no longer opens or owns the WebSocket server. Ending training does not shut down the stream server.

## Evaluation

The snapshot JSON payload format remains unchanged. Browser ping messages receive a `pong` from the stream server, and snapshot messages are relayed to all other connected clients.

## Remaining Risks

- The standalone stream server requires the `websockets` package in the active Python environment.
- The server intentionally binds only to `127.0.0.1`; remote renderer access must go through SSH port forwarding.

## 2026-07-15 Notebook Interrupt Cleanup

- The notebook Train cell now catches `KeyboardInterrupt` inside `_run_monitored_process()` and terminates the active `train.py` subprocess before re-raising the interrupt.
- Termination first calls `process.terminate()` and waits up to 10 seconds; if the process does not exit, it falls back to `process.kill()`.
- This cleanup affects the training subprocess only. The standalone stream server started by `scripts/start_trainer_stream.ps1` remains a separate process and should still be stopped with Ctrl+C in its own terminal.
