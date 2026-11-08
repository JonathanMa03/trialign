"""Loopback-only JSON/static server following Modern AutoCrit's browser pattern."""

from __future__ import annotations

import base64
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from trialign.runtime.service import ComparisonService

STATIC = Path(__file__).parent / "static"


def handler(service):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

        def dispatch(self, method):
            try:
                host = self.headers.get("Host", "")
                allowed = {
                    f"127.0.0.1:{self.server.server_port}",
                    f"localhost:{self.server.server_port}",
                }
                if host not in allowed:
                    return self.send_json(403, {"error": "Loopback host required."})
                origin = self.headers.get("Origin")
                if method == "POST" and origin and origin not in {"http://" + h for h in allowed}:
                    return self.send_json(403, {"error": "Cross-origin writes are not allowed."})
                path = urlparse(self.path).path
                body = self.body() if method == "POST" else {}
                if path == "/api/health":
                    return self.send_json(200, {"status": "ok", **service.capabilities()})
                if path == "/api/trials" and method == "GET":
                    return self.send_json(200, {"trials": service.catalog.trials()})
                if path == "/api/import" and method == "POST":
                    content = base64.b64decode(body.get("content_base64", ""), validate=True)
                    return self.send_json(
                        200, service.import_file(str(body.get("filename", "")), content)
                    )
                if path == "/api/runs":
                    return self.send_json(
                        202 if method == "POST" else 200,
                        service.create(body) if method == "POST" else {"runs": service.list_runs()},
                    )
                if path.startswith("/api/runs/"):
                    parts = path.strip("/").split("/")
                    run_id = parts[2]
                    if method == "GET" and len(parts) == 3:
                        return self.send_json(200, service.get(run_id))
                    if method == "POST" and parts[-1] == "cancel":
                        return self.send_json(200, service.cancel(run_id))
                    if method == "POST" and parts[-1] == "review":
                        return self.send_json(200, service.review(run_id, body))
                    if method == "GET" and parts[-1] == "export":
                        return self.send_json(200, service.get(run_id), download=run_id + ".json")
                if path.startswith("/api/"):
                    return self.send_json(404, {"error": "Unknown API resource."})
                if method != "GET":
                    return self.send_json(405, {"error": "Method not allowed."})
                name = "index.html" if path == "/" else path.lstrip("/")
                if name not in {"index.html", "app.js", "styles.css"}:
                    return self.send_json(404, {"error": "Not found."})
                content = (STATIC / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "text/plain")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)
            except KeyError:
                self.send_json(404, {"error": "Saved run not found."})
            except (ValueError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)[:500]})
            except Exception:
                self.send_json(500, {"error": "Request failed; saved runs are preserved."})

        def body(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 21_000_000:
                raise ValueError("Request exceeds the upload limit.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object.")
            return payload

        def send_json(self, status, payload, download=None):
            content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            return

    return Handler


def run_browser_app(root: Path, *, port=8766, open_browser=False, runtime_directory=None):
    service = ComparisonService(root, directory=runtime_directory)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(service))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"TriAlign local comparison UI: {url}", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
