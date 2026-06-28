"""
教程页面独立服务器
用法: python tutorial_server.py
访问: http://localhost:8080
"""
import os, sys
from pathlib import Path

# 定位到 static/tutorial.html 所在目录
ROOT = Path(__file__).resolve().parent / "static"
os.chdir(str(ROOT))

# 用 http.server 启动
from http.server import HTTPServer, SimpleHTTPRequestHandler

class TutorialHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # 只提供 tutorial.html
        if self.path == "/" or self.path == "/tutorial.html":
            self.path = "/tutorial.html"
            return super().do_GET()
        # 其他路径一律返回 404
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        print(f"[教程] {args[0]} {args[1]} {args[2]}")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), TutorialHandler)
    print(f"📖 教程页面: http://localhost:{port}")
    server.serve_forever()
