import sys
from pathlib import Path
import os
import numpy as np
import torch.backends.cudnn
import torch.cuda


sys.path.append(str(Path(__file__).resolve().parent) + "/../step1_preprocessing/")
from make_dataset_pereira2018 import check_sub_id


def add_optimus_path() -> None:
    sys.path.append(str(Path(__file__).resolve().parent.joinpath("optimus", "code", "examples", "big_ae")))
    sys.path.append(str(Path(__file__).resolve().parent.joinpath("optimus", "code")))


def seed_everything(seed: int = 1234):
    # random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
