"""Local SSH tunnel to gwj 0810 training TensorBoard (all stages in one TB)."""
from __future__ import annotations

import select
import socketserver
import threading
import webbrowser

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASS = "gwj@#@2026"
REMOTE_TB_HOST = "127.0.0.1"
REMOTE_TB_PORT = 6040
LOCAL_PORT = 6040


class Handler(socketserver.BaseRequestHandler):
    chain_host = REMOTE_TB_HOST
    chain_port = REMOTE_TB_PORT
    ssh_transport: paramiko.Transport | None = None

    def handle(self) -> None:
        assert self.ssh_transport is not None
        try:
            chan = self.ssh_transport.open_channel(
                "direct-tcpip",
                (self.chain_host, self.chain_port),
                self.request.getpeername(),
            )
        except Exception:
            return
        if chan is None:
            return
        try:
            while True:
                r, _, _ = select.select([self.request, chan], [], [])
                if self.request in r:
                    data = self.request.recv(1024)
                    if not data:
                        break
                    chan.send(data)
                if chan in r:
                    data = chan.recv(1024)
                    if not data:
                        break
                    self.request.send(data)
        finally:
            chan.close()
            self.request.close()


class ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport unavailable")

    Handler.ssh_transport = transport
    server = ForwardServer(("127.0.0.1", LOCAL_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{LOCAL_PORT}/"
    print(f"TensorBoard tunnel ready: {url}")
    print(f"(remote {REMOTE_TB_HOST}:{REMOTE_TB_PORT} on {HOST}, 0810 stage1+2+3)")
    print("Keep this window open. Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        while thread.is_alive():
            thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\nStopping tunnel...")
    finally:
        server.shutdown()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
