"""Local SSH tunnel for gwj TensorBoard stage1_v31 -> localhost:6029"""
import select
import socket
import sys
import threading

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASS = "gwj@#@2026"
REMOTE_PORT = 6029
LOCAL_PORT = 6029


def forward(local_sock, remote_host, remote_port, transport):
    chan = transport.open_channel(
        "direct-tcpip", (remote_host, remote_port), local_sock.getpeername()
    )

    def relay(src, dst):
        while True:
            r, _, _ = select.select([src, dst], [], [], 1.0)
            if src in r:
                data = src.recv(1024)
                if not data:
                    break
                dst.sendall(data)
            if dst in r:
                data = dst.recv(1024)
                if not data:
                    break
                src.sendall(data)
        src.close()
        dst.close()

    threading.Thread(target=relay, args=(local_sock, chan), daemon=True).start()
    threading.Thread(target=relay, args=(chan, local_sock), daemon=True).start()


def main():
    transport = paramiko.Transport((HOST, 22))
    transport.connect(username=USER, password=PASS)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", LOCAL_PORT))
    sock.listen(5)
    print(f"TUNNEL_OK http://127.0.0.1:{LOCAL_PORT}", flush=True)
    while True:
        client, _ = sock.accept()
        forward(client, "127.0.0.1", REMOTE_PORT, transport)


if __name__ == "__main__":
    main()
