import torch
import random
import numpy as np
from config import get_config
from data_loader import get_data_loader
from train import Trainer

def main(config):

    # ensure reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # instantiate train data loaders
    train_loader = get_data_loader(config=config)

    trainer = Trainer(config, train_loader=train_loader)
    trainer.train()

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)