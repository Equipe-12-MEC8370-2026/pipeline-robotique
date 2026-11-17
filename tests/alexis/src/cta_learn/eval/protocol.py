"""Local policy-server protocol: the simulator (isaacsim-env) talks to a policy running in .venv-learn.

Only stdlib + numpy, so both environments can import it. Arrays travel as raw bytes with their
dtype/shape in a JSON header (never pickled numpy objects: the two venvs have different numpy
versions). One message = 4-byte header length + JSON header + concatenated array buffers.

    server:  serve(handler, address)          handler(kind, meta, arrays) -> (meta, arrays)
    client:  with PolicyClient(address) as pc: pc.reset(); action = pc.act({"observation.state": q, ...})
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from multiprocessing.connection import Client, Listener

import numpy as np

DEFAULT_ADDRESS = ("127.0.0.1", 50765)
AUTHKEY = b"cta-learn-policy"

Arrays = dict[str, np.ndarray]


def pack(kind: str, meta: dict | None = None, arrays: Arrays | None = None) -> bytes:
    arrays = {k: np.ascontiguousarray(v) for k, v in (arrays or {}).items()}
    header = {
        "kind": kind,
        "meta": meta or {},
        "arrays": [{"name": k, "dtype": v.dtype.str, "shape": list(v.shape)} for k, v in arrays.items()],
    }
    hb = json.dumps(header).encode()
    return b"".join([struct.pack("<I", len(hb)), hb, *(v.tobytes() for v in arrays.values())])


def unpack(blob: bytes) -> tuple[str, dict, Arrays]:
    (n,) = struct.unpack_from("<I", blob, 0)
    header = json.loads(blob[4 : 4 + n])
    off, arrays = 4 + n, {}
    for spec in header["arrays"]:
        dt = np.dtype(spec["dtype"])
        size = int(np.prod(spec["shape"], dtype=np.int64)) * dt.itemsize
        arrays[spec["name"]] = np.frombuffer(blob, dt, count=size // dt.itemsize, offset=off).reshape(spec["shape"]).copy()
        off += size
    if off != len(blob):
        raise ValueError(f"message size mismatch: parsed {off} of {len(blob)} bytes")
    return header["kind"], header["meta"], arrays


Handler = Callable[[str, dict, Arrays], tuple[dict, Arrays]]


def serve(handler: Handler, address=DEFAULT_ADDRESS, max_clients: int | None = None, on_ready=None) -> None:
    """Answer requests one client at a time. ``kind == "close"`` ends a client; ``"shutdown"`` stops the server."""
    with Listener(address, authkey=AUTHKEY) as listener:
        if on_ready:
            on_ready(listener.address)
        served = 0
        while max_clients is None or served < max_clients:
            with listener.accept() as conn:
                served += 1
                while True:
                    try:
                        kind, meta, arrays = unpack(conn.recv_bytes())
                    except EOFError:
                        break
                    if kind in ("close", "shutdown"):
                        conn.send_bytes(pack("ok"))
                        if kind == "shutdown":
                            return
                        break
                    try:
                        out_meta, out_arrays = handler(kind, meta, arrays)
                        conn.send_bytes(pack("ok", out_meta, out_arrays))
                    except Exception as e:  # noqa: BLE001  report to the client instead of dying
                        conn.send_bytes(pack("error", {"error": f"{type(e).__name__}: {e}"}))


class PolicyClient:
    def __init__(self, address=DEFAULT_ADDRESS):
        self.conn = Client(tuple(address) if isinstance(address, list) else address, authkey=AUTHKEY)

    def call(self, kind: str, meta: dict | None = None, arrays: Arrays | None = None) -> tuple[dict, Arrays]:
        self.conn.send_bytes(pack(kind, meta, arrays))
        rkind, rmeta, rarrays = unpack(self.conn.recv_bytes())
        if rkind == "error":
            raise RuntimeError(f"policy server: {rmeta['error']}")
        return rmeta, rarrays

    def info(self) -> dict:
        return self.call("info")[0]

    def reset(self) -> None:
        self.call("reset")

    def act(self, obs: Arrays) -> np.ndarray:
        return self.call("act", arrays=obs)[1]["action"]

    def close(self, shutdown: bool = False) -> None:
        try:
            self.call("shutdown" if shutdown else "close")
        finally:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
