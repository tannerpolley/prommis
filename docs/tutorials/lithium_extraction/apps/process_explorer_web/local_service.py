"""Serve Paper 2 campaign results and POST /api/campaign/solve on loopback port 8766."""

import json
import os
import signal
import subprocess
import sys
import uuid
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
PORT = int(os.environ.get("PORT", "8766"))
SOLVE_LOCK = threading.Lock()
def solve_campaign(params: dict) -> dict:
    """Run the campaign worker with its own decisions, timer and acceptance checks."""
    from apps.process_explorer_web.campaign import SOLVENT_KEYS, SCOPE_KEYS, reason
    from analyses.prototypes.modular_process.scripts.run_campaign import jobs

    if params.get("solvent") not in ("set_a", "set_c"):
        raise ValueError("Live calculation is available only for HBTA/TOPO.")
    phase = params.get("phase", "selected")
    if phase not in ("fixed", "selected"):
        raise ValueError("Choose fixed design or optimization.")
    stages = params.get("stages")
    if not isinstance(stages, int) or isinstance(stages, bool) or not 1 <= stages <= 5:
        raise ValueError("Choose one to five extraction stages.")
    end = params.get("end", "low") if params["solvent"] == "set_c" else "fitted"
    job = next((j for j in jobs() if j["solvent"] == SOLVENT_KEYS[params["solvent"]]
                and j["site"] == params.get("site") and j["scope"] == SCOPE_KEYS.get(params.get("scope"))
                and j["case"] == params.get("case") and j["stages"] == stages and j["end"] == end), None)
    if job is None:
        raise ValueError("Unknown solvent, produced water, scope, case or calibration range.")
    job = dict(job, phase=phase, parent_id=job["id"], id="app_" + uuid.uuid4().hex)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="li_campaign_") as folder:
        output = Path(folder) / "result.json"
        log = ROOT / "build/modular_campaign" / job["id"] / "worker.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open("w") as stream:
                command = [sys.executable, str(HERE / "campaign.py"), json.dumps(job), str(output)]
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=max(0, 290 - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    if hasattr(os, 'killpg'):  # the worker and its IPOPT processes
                        os.killpg(process.pid, signal.SIGKILL)
                    else:  # Windows: the worker only
                        process.kill()
                    process.wait()
                    raise
                if code:
                    raise subprocess.CalledProcessError(code, command)
            result = json.loads(output.read_text())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError, ValueError) as error:
            result = {"candidate": job, "status": "unresolved", "reason": str(error),
                      "elapsed_wall_s": time.monotonic() - started}
    if result["status"] not in ("optimal", "grid"):
        result["campaign_reason"] = result["reason"]
        result["reason"] = reason([result["reason"]])
    return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def translate_path(self, path):
        # Shared helpers live in apps/common_web; git symlinks do not survive core.symlinks=false checkouts.
        if path.startswith("/common/"):
            self.directory, path = str(HERE.parent / "common_web"), path[len("/common"):]
        elif path.startswith("/map/"):  # the retained #245 solvent–water-chemistry map figure
            self.directory, path = str(ROOT / "analyses/prototypes/solvent_water_chemistry_map/figures"), path[len("/map"):]
        return super().translate_path(path)

    def end_headers(self):
        if self.path.startswith("/api/"):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/health":
            return self._json(200, {"ok": True})
        return super().do_GET()

    def do_POST(self):
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/api/campaign/solve":
            return self._json(404, {"error": "unknown endpoint"})
        if not SOLVE_LOCK.acquire(blocking=False):
            return self._json(409, {"error": "A calculation is already running; try again when it finishes."})
        try:
            return self._json(200, solve_campaign(json.loads(data or b"{}")))
        except Exception as error:  # report the solver's own message; it is not a statement about the chemistry
            return self._json(422, {"error": f"{type(error).__name__}: {error}"})
        finally:
            SOLVE_LOCK.release()


if __name__ == "__main__":
    print(f"Process explorer: http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
