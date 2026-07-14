"""Local WebSocket broadcaster for live renderer snapshots."""

from __future__ import annotations

import json
import threading
from typing import Any


class TrainerWebSocketServer:
    """Broadcast trainer snapshots to renderer clients over a local WebSocket."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        if host != "127.0.0.1":
            raise ValueError("TrainerWebSocketServer must bind only to 127.0.0.1.")
        if not 1 <= int(port) <= 65535:
            raise ValueError("TrainerWebSocketServer port must be between 1 and 65535.")
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
        self._thread = threading.Thread(target=self._serve, name="osn-gs-ws-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Timed out while starting the trainer WebSocket server.")
        if self._server is None:
            raise RuntimeError(f"Could not start trainer WebSocket server on {self.host}:{self.port}.")
        print(f"[WS] trainer server listening on ws://{self.host}:{self.port}", flush=True)

    def stop(self) -> None:
        server = self._server
        if server is not None:
            server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._server = None

    def broadcast(self, payload: str) -> int:
        with self._lock:
            clients = tuple(self._clients)
        delivered = 0
        for client in clients:
            try:
                client.send(payload)
                delivered += 1
            except Exception:
                with self._lock:
                    self._clients.discard(client)
        return delivered

    def _serve(self) -> None:
        try:
            from websockets.sync.server import serve

            with serve(self._handle_client, self.host, self.port, max_size=None) as server:
                self._server = server
                self._ready.set()
                server.serve_forever()
        finally:
            self._ready.set()

    def _handle_client(self, websocket: Any) -> None:
        with self._lock:
            self._clients.add(websocket)
        try:
            websocket.send(json.dumps({
                "type": "hello",
                "message": "Connected to OSN-GS trainer WebSocket server.",
            }))
            for message in websocket:
                try:
                    parsed = json.loads(message)
                    if parsed.get("type") == "ping":
                        websocket.send(json.dumps({"type": "pong", "source": "trainer"}))
                except (TypeError, ValueError):
                    pass
        finally:
            with self._lock:
                self._clients.discard(websocket)
