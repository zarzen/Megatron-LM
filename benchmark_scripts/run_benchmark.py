import argparse
from ssh_comm import get_host_ips, init_ssh_clients, parallel_exec_diff_cmd_wait


def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostfile", required=True)
    parser.add_argument("--tensor-mp-size", type=int)
    parser.add_argument("--pipeline-mp-size", type=int)
    parser.add_argument("--global-bs", default=8192, type=int)
    parser.add_argument("--timeout", default=1800, help="timeout in seconds", type=int)
    return parser.parse_args()

def main():
    """"""
    args = get_args()

    host_ips = get_host_ips(args.hostfile)
    ssh_clients = init_ssh_clients(host_ips)

    master_ip = host_ips[0]
    nnodes = len(host_ips)
    TMP = args.tensor_mp_size
    PMP = args.pipeline_mp_size

    node_cmds = []
    for node_rank in range(nnodes):
        cmd = " && ".join([
            "cd /tmp/Megatron-LM",
            f"./benchmark_scripts/bench_pretrain_bert.sh {master_ip} {nnodes} {node_rank} {TMP} {PMP}"
        ])
        node_cmds.append(cmd)
    
    parallel_exec_diff_cmd_wait(ssh_clients, node_cmds, timeout=args.timeout)

if __name__ == "__main__":
    main()