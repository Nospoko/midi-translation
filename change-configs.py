import glob

import torch
from omegaconf import OmegaConf
import os
from evals import load_checkpoint

def main():
    for path in glob.glob('models/*.pt'):
        torch.load(path)
        run_name = 'midi-transformer-' + '-'.join(path.split('-')[-6:-1])
        print(run_name)
        checkpoint = load_checkpoint(run_name)
        checkpoint['cfg']['dataset']['dataset_name'] = 'roszcz/maestro-v1'

        torch.save(checkpoint, path)



if __name__ == '__main__':
    main()