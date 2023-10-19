import hydra
from argparse import ArgumentParser
import torch
from torch.cuda import device_count
from torch.multiprocessing import spawn
from learner import mytrain
from params import params

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


@hydra.main(config_path="conf", config_name="conf")
def main(args):
    torch.manual_seed(args.seed)
    replica_count = device_count()
    # replica_count=1
    if replica_count > 1:
        if params.batch_size % replica_count != 0:
            raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
        params.batch_size = params.batch_size // replica_count
        port = _get_free_port()
        spawn(mytrain, args=(replica_count, port, args), nprocs=replica_count, join=True)
    else:
        mytrain(0, 0, 0, args)


if __name__ == '__main__':
    main()
   