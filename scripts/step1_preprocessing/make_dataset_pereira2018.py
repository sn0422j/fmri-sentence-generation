import argparse
import os
import pickle
from configparser import ConfigParser
from pathlib import Path
from typing import Final

import numpy as np
import scipy.io
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit


def check_sub_id(exp_id: int, sub_id: int):
    sub_id_for_exp_2 = [0, 2, 4, 7, 8, 9, 14, 15]  # P01 is sub_id=0
    sub_id_for_exp_3 = [0, 2, 3, 4, 7, 15]
    if exp_id == 2:
        if sub_id not in sub_id_for_exp_2:
            raise ValueError(f"sub_id {sub_id} is not in the exp 2.")
    elif exp_id == 3:
        if sub_id not in sub_id_for_exp_3:
            raise ValueError(f"sub_id {sub_id} is not in the exp 3.")
    else:
        raise ValueError(f"{exp_id=} not found.")


def load_volume(main_dir_path: str, sub_name: str = "P01", exp_id: int = 0):
    if exp_id == 2:
        number_items = 384
    elif exp_id == 3:
        number_items = 243
    else:
        raise ValueError(f"{exp_id=} not found.")
    file_path = os.path.join(main_dir_path, f"{sub_name}/{sub_name}/examples_{number_items}sentences.mat")
    matfile = scipy.io.loadmat(file_path)

    labels = matfile["labelsSentences"].copy()  # Sentence labels number.
    labels = np.hstack([labels, matfile["keySentences"]])  # Sentence texts.
    labels = np.hstack([labels, matfile["labelsPassageForEachSentence"]])  # Passage labels number.

    passage_category = matfile["labelsPassageCategory"]
    category_name = matfile["keyPassageCategory"]

    return matfile["examples"], labels, passage_category, category_name


def train_test_split(labels: np.ndarray, passage_category: np.ndarray, exp_id: int = 0):
    if exp_id == 2:
        test_size = 4  # 4 of 24 topics (16 of 96 passages, 48 of 384 sentences) selected for test.
        train_size = 20
    elif exp_id == 3:
        test_size = 4  # 4 of 24 topics (12 of 72 passages, 36-48 of 243 sentences) selected for test.
        train_size = 20
    else:
        raise ValueError(f"{exp_id=} not found.")

    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, train_size=train_size, random_state=1234)

    index_dict = {}
    for cv, (train_passage_index, test_passage_index) in enumerate(
        splitter.split(passage_category, groups=passage_category)
    ):
        # train_passage = passage_category[train_passage_index]
        # test_passage = passage_category[test_passage_index]
        train_passage_index += 1
        test_passage_index += 1  # Change to matlab index
        train_index = []
        test_index = []
        train_passage = []
        test_passage = []
        for i, label in enumerate(labels):
            # label[0] is sentence_index (384 or 243 passages)
            # label[1] is presented_sentence
            # label[2] is passage_index (96 or 72 passages)
            if label[2] in test_passage_index:
                test_index.append(i)
                test_passage.append(passage_category[label[2] - 1, 0])
            elif label[2] in train_passage_index:
                train_index.append(i)
                train_passage.append(passage_category[label[2] - 1, 0])
            else:
                assert ValueError(f"train_test_split: index {i} not found.")
        index_dict[cv] = {
            "train_passage": train_passage,
            "test_passage": test_passage,
            "train_index": train_index,
            "test_index": test_index,
        }

    return index_dict


def check_index_dict(index_dict: dict, timeseries: np.ndarray, labels: np.ndarray, cv: int = 1):
    train_index = index_dict[cv]["train_index"]
    test_index = index_dict[cv]["test_index"]
    train_timeseries, train_labels = timeseries[train_index], labels[train_index]
    test_timeseries, test_labels = timeseries[test_index], labels[test_index]
    print(f"check cv{cv}:")
    print("  train data shape:", train_timeseries.shape)
    print("  test data shape:", test_timeseries.shape)
    print("  train label shape:", train_labels.shape)
    print("  test label shape:", test_labels.shape)
    print("  train passages:", np.unique(index_dict[cv]["train_passage"], return_counts=True))
    print("  test passages:", np.unique(index_dict[cv]["test_passage"], return_counts=True))


def main():
    config_ini = ConfigParser()
    config_ini.read("config.ini", encoding="utf-8")
    MAIN_DIR_PATH: Final[str] = config_ini.get("Pereira2018", "main_directory")

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", help="experiment id from [2,3]", type=int)
    parser.add_argument("sub_id", help="subject id (P01 is sub_id=0)", type=int)
    args = parser.parse_args()
    print(args)

    check_sub_id(exp_id=args.exp_id, sub_id=args.sub_id)
    if args.sub_id == 0:
        sub_name = "P01"
    else:
        sub_name = "M{args.sub_id:02}"

    # load timeseries and caption, and preprocess them.
    timeseries, labels, passage_category, category_name = load_volume(
        MAIN_DIR_PATH, sub_name=sub_name, exp_id=args.exp_id
    )

    # Split data into train and test.
    index_dict = train_test_split(labels, passage_category, exp_id=args.exp_id)
    check_index_dict(index_dict, timeseries, labels)

    # Save timeseries in .npy file
    SAVE_DIR_PATH: Final[str] = f"./data/exp{args.exp_id}/s{args.sub_id:02}"
    os.makedirs(SAVE_DIR_PATH, exist_ok=True)
    save_file_path = Path(SAVE_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_example.npy")
    np.save(save_file_path, timeseries)

    # Save labels in .mat file
    save_file_path = Path(SAVE_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_label.mat")
    scipy.io.savemat(
        save_file_path,
        {
            "labels": labels,
            "category_name": category_name,
        },
    )
    save_file_path = Path(SAVE_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_index.pickle")
    with open(save_file_path, "wb") as f:
        pickle.dump(index_dict, f)


if __name__ == "__main__":
    main()
