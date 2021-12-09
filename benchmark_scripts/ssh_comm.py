from pssh.clients.native.single import SSHClient
import time
from termcolor import colored
import threading
from typing import List


def get_host_ips(hostfile):
    host_ips = []
    host_slots = []
    with open(hostfile) as in_file:
        for line in in_file:
            if line.strip() != "":
                ip, slots = [v.strip() for v in line.split(' ')]
                host_ips += [ip]
                host_slots += [int(slots.split('=')[1])]
    return host_ips

def init_ssh_clients(host_ips):
    """"""
    clients = []
    for ip in host_ips:
        cli = SSHClient(ip) # assume password less login
        clients.append(cli)
        print(f'connected to {ip}')
    return clients

def __run_cmd(cli: SSHClient, cmd: str, timeout):
    output = cli.run_command(cmd, use_pty=True, read_timeout=timeout)
    print(f"{output.host} :: {cmd}")
    read_start_time = time.time()
    for line in output.stdout:
        print(f"{output.host} :: {line}")
    for line in output.stderr:
        print(colored(f'{output.host}::{line}', 'red'))
    # after read timeout
    if time.time() - read_start_time > timeout:
        # close channel to exit
        cli.close_channel()

def parallel_exec_wait(clients, cmd: str, timeout=600): 
    """ default timeout 10min
    """
    cli_thds = []
    for c in clients:
        t = threading.Thread(target=__run_cmd, args=(c, cmd, timeout))
        t.start()
        cli_thds.append(t)

    for _, t in enumerate(cli_thds):
        t.join()

def parallel_exec_diff_cmd_wait(clients: List[SSHClient], cmds: List[str], timeout=600): 
    """ default timeout 10min
    """
    assert len(clients) == len(cmds)

    cli_thds = []
    for cli, cmd in zip(clients, cmds):
        t = threading.Thread(target=__run_cmd, args=(cli, cmd, timeout))
        t.start()
        cli_thds.append(t)

    for _, t in enumerate(cli_thds):
        t.join()