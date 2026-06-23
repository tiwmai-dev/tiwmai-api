from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def _send_empty_response(self):
        self.send_response(404)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._send_empty_response()

    def do_POST(self):
        self._send_empty_response()

    def do_PUT(self):
        self._send_empty_response()

    def do_PATCH(self):
        self._send_empty_response()

    def do_DELETE(self):
        self._send_empty_response()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
