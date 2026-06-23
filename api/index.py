import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def _send_disabled_response(self):
        body = json.dumps(
            {
                "status": "disabled",
                "message": "This production API deployment is intentionally disabled.",
            }
        ).encode("utf-8")

        self.send_response(404)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send_disabled_response()

    def do_POST(self):
        self._send_disabled_response()

    def do_PUT(self):
        self._send_disabled_response()

    def do_PATCH(self):
        self._send_disabled_response()

    def do_DELETE(self):
        self._send_disabled_response()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
