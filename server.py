import os
import sys
import socket
import threading
import http.server
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
import zipfile
import io
import urllib.parse

if getattr(sys, 'frozen', False):
    Www_DIR = sys._MEIPASS
else:
    Www_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = ['happy', 'sad']
EXAMPLE_DIRS = ['train_examples', 'test_examples']


def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip and not ip.startswith('127.'):
                return ip
        except Exception:
            pass
        return None


def ensure_dirs():
    results = []
    for example_dir in EXAMPLE_DIRS:
        dir_path = os.path.join(Www_DIR, example_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            results.append(f'创建目录: {example_dir}/')
        for cat in CATEGORIES:
            cat_path = os.path.join(dir_path, cat)
            if not os.path.exists(cat_path):
                os.makedirs(cat_path, exist_ok=True)
                results.append(f'创建目录: {example_dir}/{cat}/')
    return results


_bundle_cache = {}


class COOPHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/bundle':
            qs = urllib.parse.parse_qs(parsed.query)
            dir_param = qs.get('dir', [''])[0]
            if not dir_param:
                self.send_error(400, 'Missing dir parameter')
                return
            safe_dir = dir_param.replace('..', '').strip('/')
            real_dir = os.path.join(Www_DIR, safe_dir)
            real_dir = os.path.normpath(real_dir)
            if not real_dir.startswith(os.path.normpath(Www_DIR)):
                self.send_error(403, 'Forbidden')
                return
            if not os.path.isdir(real_dir):
                self.send_error(404, 'Directory not found')
                return

            cache_key = safe_dir
            cached = _bundle_cache.get(cache_key)
            if cached:
                data, count = cached
            else:
                buf = io.BytesIO()
                count = 0
                with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as zf:
                    for fname in sorted(os.listdir(real_dir)):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                            fpath = os.path.join(real_dir, fname)
                            zf.write(fpath, fname)
                            count += 1
                data = buf.getvalue()
                _bundle_cache[cache_key] = (data, count)

            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-File-Count', str(count))
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
            self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
            self.end_headers()
            self.wfile.write(data)
            return

        super().do_GET()

    def translate_path(self, path):
        import urllib.parse
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        path = urllib.parse.unquote(path)
        path = os.path.normpath(path)
        words = path.split(os.sep)
        words = [w for w in words if w]
        path = Www_DIR
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            path = os.path.join(path, word)
        return path

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        cache_path = self.translate_path(self.path)
        _, ext = os.path.splitext(cache_path)
        ext = ext.lower()
        if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'):
            self.send_header('Cache-Control', 'public, max-age=86400')
        elif ext in ('.json', '.onnx', '.wasm', '.mjs'):
            self.send_header('Cache-Control', 'public, max-age=3600')
        elif ext in ('.js', '.css'):
            self.send_header('Cache-Control', 'public, max-age=300')
        super().end_headers()

    def log_message(self, format, *args):
        pass


class ServerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('情绪分类器 - Web 服务器')
        self.root.resizable(False, False)
        self.server = None
        self.server_thread = None
        self.running = False

        self._build_ui()
        self.root.after(300, self._auto_start)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text='情绪分类器 Web 服务器',
                  font=('Microsoft YaHei UI', 14, 'bold')).pack(pady=(0, 12))

        port_frame = ttk.Frame(main)
        port_frame.pack(fill=tk.X, pady=4)
        ttk.Label(port_frame, text='端口号:').pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value='5000')
        self.port_entry = ttk.Entry(port_frame, textvariable=self.port_var, width=8)
        self.port_entry.pack(side=tk.LEFT, padx=6)

        self.start_btn = ttk.Button(port_frame, text='启动服务器', command=self._toggle_server)
        self.start_btn.pack(side=tk.LEFT, padx=6)
        self.open_btn = ttk.Button(port_frame, text='打开浏览器', command=self._open_browser, state=tk.DISABLED)
        self.open_btn.pack(side=tk.LEFT, padx=6)

        sep = ttk.Separator(main, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=8)

        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=4)
        ttk.Label(status_frame, text='服务器状态:').pack(side=tk.LEFT)
        self.status_indicator = tk.Canvas(status_frame, width=16, height=16, highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=6)
        self.status_dot = self.status_indicator.create_oval(2, 2, 14, 14, fill='#999999', outline='#666666')
        self.status_label = ttk.Label(status_frame, text='未启动', foreground='#666666')
        self.status_label.pack(side=tk.LEFT)

        coop_frame = ttk.Frame(main)
        coop_frame.pack(fill=tk.X, pady=4)
        ttk.Label(coop_frame, text='Cross-Origin Isolation:').pack(side=tk.LEFT)
        self.coop_indicator = tk.Canvas(coop_frame, width=16, height=16, highlightthickness=0)
        self.coop_indicator.pack(side=tk.LEFT, padx=6)
        self.coop_dot = self.coop_indicator.create_oval(2, 2, 14, 14, fill='#999999', outline='#666666')
        self.coop_label = ttk.Label(coop_frame, text='未检测', foreground='#666666')
        self.coop_label.pack(side=tk.LEFT)

        lan_frame = ttk.Frame(main)
        lan_frame.pack(fill=tk.X, pady=4)
        ttk.Label(lan_frame, text='局域网访问:').pack(side=tk.LEFT)
        self.lan_label = ttk.Label(lan_frame, text='未启动', foreground='#666666', font=('Consolas', 9))
        self.lan_label.pack(side=tk.LEFT, padx=6)

        sep2 = ttk.Separator(main, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, pady=8)

        ttk.Label(main, text='启动日志:', font=('Microsoft YaHei UI', 9, 'bold')).pack(anchor=tk.W)
        log_frame = ttk.Frame(main)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log_text = tk.Text(log_frame, height=12, width=60, font=('Consolas', 9),
                                state=tk.DISABLED, wrap=tk.WORD, bg='#1e1e1e', fg='#d4d4d4')
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _log(self, msg, tag=None):
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, msg + '\n', tag)
        else:
            self.log_text.insert(tk.END, msg + '\n')
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_status(self, running):
        if running:
            self.status_indicator.itemconfig(self.status_dot, fill='#22c55e', outline='#16a34a')
            self.status_label.configure(text=f'运行中 (端口 {self.port_var.get()})', foreground='#16a34a')
            self.start_btn.configure(text='停止服务器')
            self.port_entry.configure(state=tk.DISABLED)
            self.open_btn.configure(state=tk.NORMAL)
        else:
            self.status_indicator.itemconfig(self.status_dot, fill='#999999', outline='#666666')
            self.status_label.configure(text='已停止', foreground='#666666')
            self.start_btn.configure(text='启动服务器')
            self.port_entry.configure(state=tk.NORMAL)
            self.open_btn.configure(state=tk.DISABLED)

    def _set_coop_status(self, enabled):
        if enabled:
            self.coop_indicator.itemconfig(self.coop_dot, fill='#22c55e', outline='#16a34a')
            self.coop_label.configure(text='已启用 (COOP + COEP)', foreground='#16a34a')
        else:
            self.coop_indicator.itemconfig(self.coop_dot, fill='#ef4444', outline='#dc2626')
            self.coop_label.configure(text='未启用', foreground='#dc2626')

    def _check_dirs_and_json(self):
        self._log('检查目录结构...')
        results = ensure_dirs()
        for r in results:
            self._log('  ' + r)

    def _toggle_server(self):
        if self.running:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        port_str = self.port_var.get().strip()
        try:
            port = int(port_str)
            if port < 1 or port > 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror('端口错误', f'无效端口号: {port_str}\n请输入 1-65535 之间的整数。')
            return

        self._check_dirs_and_json()

        try:
            self.server = http.server.ThreadingHTTPServer(('0.0.0.0', port), COOPHandler)
        except OSError as e:
            messagebox.showerror('启动失败', f'无法启动服务器:\n{e}')
            self._log(f'启动失败: {e}')
            return

        self.running = True
        self._set_status(True)
        self._set_coop_status(True)
        self._log(f'Web 服务器已启动: http://localhost:{port}')
        self._log('监听地址: 0.0.0.0 (所有网络接口)')

        lan_ip = get_lan_ip()
        if lan_ip:
            self.lan_label.configure(text=f'http://{lan_ip}:{port}', foreground='#3b82f6')
            self._log(f'局域网访问: http://{lan_ip}:{port}')
        else:
            self.lan_label.configure(text='无法获取局域网IP', foreground='#f59e0b')
            self._log('无法获取局域网 IP 地址')

        self._log('Cross-Origin Isolation 头已配置:')
        self._log('  Cross-Origin-Opener-Policy: same-origin')
        self._log('  Cross-Origin-Embedder-Policy: require-corp')
        self._log('  Cross-Origin-Resource-Policy: cross-origin')

        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.root.after(500, lambda: webbrowser.open(f'http://localhost:{port}'))

    def _stop_server(self):
        if self.server:
            server = self.server
            self.server = None
            self.running = False
            self._set_status(False)
            self._set_coop_status(False)
            self.lan_label.configure(text='未启动', foreground='#666666')
            self._log('正在停止 Web 服务器...')

            def do_shutdown():
                try:
                    server.shutdown()
                    server.server_close()
                except Exception:
                    pass

            threading.Thread(target=do_shutdown, daemon=True).start()
        else:
            self.running = False
            self._set_status(False)
            self._set_coop_status(False)

    def _open_browser(self):
        port = self.port_var.get().strip()
        webbrowser.open(f'http://localhost:{port}')

    def _auto_start(self):
        self._log('自动启动 Web 服务器...')
        self._start_server()

    def on_close(self):
        if self.running and self.server:
            server = self.server
            self.server = None
            self.running = False
            self._set_status(False)
            self._log('正在关闭服务器...')

            def do_shutdown():
                try:
                    server.timeout = 1
                    server.shutdown()
                    server.server_close()
                except Exception:
                    pass
                self.root.after(0, self._finish_close)

            shutdown_thread = threading.Thread(target=do_shutdown, daemon=True)
            shutdown_thread.start()
            self.root.after(2000, self._force_close, shutdown_thread)
        else:
            self.root.destroy()

    def _finish_close(self):
        self._log('服务器已关闭')
        try:
            self.root.destroy()
        except Exception:
            pass

    def _force_close(self, shutdown_thread):
        if shutdown_thread.is_alive():
            self._log('强制关闭服务器')
            try:
                import os
                os._exit(0)
            except Exception:
                pass


def main():
    root = tk.Tk()
    app = ServerApp(root)
    root.protocol('WM_DELETE_WINDOW', app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
