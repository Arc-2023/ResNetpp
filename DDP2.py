import os

import torch.distributed as dist
import torch.multiprocessing as mp
from DDP import train_func





def cleanup():
    dist.destroy_process_group()


def run_demo(world_size):
    mp.spawn(train_func,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    # ray.init(include_dashboard=False, dashboard_host='0.0.0.0',ignore_reinit_error=True)
    #
    # scaling_config = ScalingConfig(num_workers=1, use_gpu=True)
    # torch_config = ray.train.torch.TorchConfig(backend='gloo')
    # trainer = ray.train.torch.TorchTrainer(train_func, train_loop_config=args, scaling_config=scaling_config,
    #                                        torch_config=torch_config)
    # result = trainer.fit()
    run_demo(1)
