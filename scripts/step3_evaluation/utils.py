import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent) + "/../step1_preprocessing/")
from make_dataset_pereira2018 import check_sub_id
