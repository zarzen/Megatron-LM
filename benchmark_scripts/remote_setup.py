import argparse
from ssh_comm import get_host_ips, init_ssh_clients, parallel_exec_wait

def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostfile", required=True)

    return parser.parse_args()

def main():
    """"""
    args = get_args()

    host_ips = get_host_ips(args.hostfile)
    ssh_clients = init_ssh_clients(host_ips)

    # clone the repo into /tmp/Megatron-LM 
    # try remove first
    parallel_exec_wait(ssh_clients, "rm -rf /tmp/Megatron-LM", 120)
    parallel_exec_wait(ssh_clients, "pip3 install pybind11")
    parallel_exec_wait(ssh_clients, "git clone -b zhen/benchmark https://github.com/zarzen/Megatron-LM.git /tmp/Megatron-LM")


if __name__ == "__main__":
    main()