from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer

from datetime import datetime

import os
import random
import numpy as np
import torch
import torch.cuda
import torch.backends.cudnn

import hydra

def make_deterministic(seed=0):
    '''
    Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    seed=int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# change here for other subjects
@hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/377/', config_name='single_gpu_hash')

def main(cfg):
    #make_deterministic()
    model = create_network(cfg)
    optimizer = create_optimizer(cfg, model)
    trainer = create_trainer(cfg, model, optimizer)
    train_loader = create_dataloader(cfg, 'train')

    # estimate start epoch
    tic = datetime.now()
    epoch = trainer.iter // len(train_loader) + 1
    while not trainer.iter > cfg.train.maxiter:
        trainer.train(epoch=epoch, train_dataloader=train_loader)
        epoch += 1

    trainer.finalize()
    toc = datetime.now()
    print('# # '*20)
    print('Elapsed time: %f seconds' % (toc-tic).total_seconds())

if __name__ == '__main__':
    main()
