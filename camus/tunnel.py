"""Host-only loopback SSH tunnel. Password stays in memory/environment."""
from __future__ import annotations
import argparse
import getpass
import os
import select
import socketserver
import paramiko


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--host',default='192.168.2.99')
    p.add_argument('--user',default='xilinx');p.add_argument('--local-port',type=int,default=8091)
    p.add_argument('--remote-port',type=int,default=8090);a=p.parse_args(argv)
    client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password=os.environ.get('PLD_BOARD_PASS') or getpass.getpass('Board SSH password: ')
    client.connect(a.host,username=a.user,password=password,timeout=10,allow_agent=False,look_for_keys=False)
    transport=client.get_transport();transport.set_keepalive(15)
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            channel=transport.open_channel('direct-tcpip',('127.0.0.1',a.remote_port),self.client_address,timeout=10)
            if channel is None:return
            try:
                while transport.is_active():
                    readers,_,_=select.select([self.request,channel],[],[],30)
                    for source in readers:
                        data=source.recv(65536)
                        if not data:return
                        (channel if source is self.request else self.request).sendall(data)
            except (OSError,EOFError):pass
            finally:channel.close()
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address=True;daemon_threads=True
    try:
        with Server(('127.0.0.1',a.local_port),Handler) as server:
            print(f'CAMUS_BOARD_TUNNEL http://127.0.0.1:{a.local_port} -> board:{a.remote_port}',flush=True)
            server.serve_forever(poll_interval=.5)
    finally:client.close()


if __name__=='__main__':main()
