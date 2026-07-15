"""Standalone local WebSocket stream server for OSN-GS snapshots."""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any


class TrainingStreamServer:
    """Accept trainer snapshots and broadcast them to renderer clients."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        if host != "127.0.0.1":
            raise ValueError("TrainingStreamServer must bind only to 127.0.0.1. Use SSH port forwarding for remote renderers.")
        if not 1 <= int(port) <= 65535:
            raise ValueError("TrainingStreamServer port must be between 1 and 65535.")
        self.host = host
        self.port = int(port)
        self._clients: set[Any] = set()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._server: Any | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, name="osn-gs-stream-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Timed out while starting the OSN-GS stream server.")
        if self._server is None:
            raise RuntimeError(f"Could not start OSN-GS stream server on {self.host}:{self.port}.")

    def serve_forever(self) -> None:
        from websockets.sync.server import serve

        with serve(self._handle_client, self.host, self.port, max_size=None) as server:
            self._server = server
            self._ready.set()
            print(f"[WS] stream server listening on ws://{self.host}:{self.port}", flush=True)
            server.serve_forever()

    def stop(self) -> None:
        server = self._server
        if server is not None:
            server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._server = None

    def broadcast(self, payload: str, sender: Any | None = None) -> int:
        with self._lock:
            clients = tuple(self._clients)
        delivered = 0
        for client in clients:
            if client is sender:
                continue
            try:
                client.send(payload)
                delivered += 1
            except Exception:
                with self._lock:
                    self._clients.discard(client)
        return delivered

    def _serve(self) -> None:
        try:
            self.serve_forever()
        finally:
            self._ready.set()

    def _handle_client(self, websocket: Any) -> None:
        with self._lock:
            self._clients.add(websocket)
        try:
            websocket.send(json.dumps({
                "type": "hello",
                "message": "Connected to OSN-GS local stream server.",
            }))
            for message in websocket:
                self._handle_message(websocket, message)
        finally:
            with self._lock:
                self._clients.discard(websocket)

    def _handle_message(self, websocket: Any, message: Any) -> None:
        if not isinstance(message, str):
            return
        try:
            parsed = json.loads(message)
        except (TypeError, ValueError):
            return

        message_type = parsed.get("type", "snapshot")
        if message_type == "ping":
            websocket.send(json.dumps({"type": "pong", "source": "stream_server"}))
            return
        if message_type == "snapshot":
            payload = json.dumps(parsed, separators=(",", ":"))
            delivered = self.broadcast(payload, sender=websocket)
            print(
                f"[WS] relayed iteration {parsed.get('iteration', '?')} to {delivered} client(s)",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local OSN-GS WebSocket stream server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Must be 127.0.0.1; use SSH port forwarding remotely.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port for trainer and renderer WebSocket clients.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = TrainingStreamServer(host=args.host, port=args.port)
    server.serve_forever()


TrainerWebSocketServer = TrainingStreamServer


if __name__ == "__main__":
    main()

