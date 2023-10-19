# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
   