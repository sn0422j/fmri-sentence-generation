import argparse
import os
import pickle
import warnings
from configparser import ConfigParser
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm
from train_pred import predict_latent
from utils import check_sub_id

warnings.simplefilter("ignore")


def load_GloVe(main_dir_path: str, exp_id: int = 0) -> np.ndarray:
    if exp_id == 2:
        number_items = 384
    elif exp_id == 3:
        number_items = 243
    else:
        raise ValueError(f"{exp_id=} not found.")
    file_path = os.path.join(
        main_dir_path, f"vectors_{number_items}sentences_dereferencedpronouns.GV42B300.average.txt"
    )
    df = pd.read_table(file_path, header=None, delim_whitespace=True)
    return df.values


def run_optimus_reconst(
    pred_latent_z: np.ndarray,
    test_text: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    import torch
    from optimus import prepare_vae, text_from_latent_code

    def run_text_from_latent_code(pred_latent_z, model_vae, tokenizer_decoder, args):
        text_reconst_list = []
        for latent_z in tqdm(pred_latent_z):
            latent_z = torch.from_numpy(latent_z.reshape(1, -1).copy()).to(args.device, dtype=torch.float)
            text_reconst = text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder)
            text_reconst_list.append(text_reconst)
        return np.array(text_reconst_list)

    model_vae, _, tokenizer_decoder, args = prepare_vae(args)
    pred_text_reconst = run_text_from_latent_code(pred_latent_z, model_vae, tokenizer_decoder, args)
    print("output example: ", "\n -> ".join([test_text[0], pred_text_reconst[0]]))
    return pred_text_reconst


def main():
    config_ini = ConfigParser()
    config_ini.read("config.ini", encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", help="experiment id from [2,3]", type=int)
    parser.add_argument("sub_id", help="subject id (P01 is sub_id=0)", type=int)
    parser.add_argument("model", help="language model name", type=str)
    parser.add_argument("--reshape", help="(option) flag of reshaping tensor", action="store_true")
    parser.add_argument("--feature_selection", help="(option) feature selection method", default="none")
    parser.add_argument("--regressor", help="(option) regressor", default="ridge")
    parser.add_argument("--debug", help="(option) flag of debug", action="store_true")
    parser.add_argument("--permute", help="(option) flag of permute test", action="store_true")
    args = parser.parse_args()
    print(args)
    NUM_FOR_DEBUG: int = 90 # In debug, we use 90 samples and run 1st fold.

    check_sub_id(exp_id=args.exp_id, sub_id=args.sub_id)

    # Load fmri data in .npy file
    LOAD_DIR_PATH: Final[str] = f"./data/exp{args.exp_id}/s{args.sub_id:02}"
    load_file_path = Path(LOAD_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_example.npy")
    fmri_data = np.load(load_file_path)
    if args.debug:
        fmri_data = fmri_data[:NUM_FOR_DEBUG]

    # Load labels in .mat file
    load_file_path = Path(LOAD_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_label.mat")
    labels = scipy.io.loadmat(load_file_path)["labels"]
    if args.debug:
        labels = labels[:NUM_FOR_DEBUG]
    load_file_path = Path(LOAD_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_index.pickle")
    with open(load_file_path, "rb") as f:
        index_dict = pickle.load(f)

    # Load latent z
    SAVE_DIR_PATH: Final[str] = f"./results/{args.model}/exp{args.exp_id}/s{args.sub_id:02}"
    os.makedirs(SAVE_DIR_PATH, exist_ok=True)
    if args.model == "optimus":
        load_file_path = Path(SAVE_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_latent.mat")
        matfile = scipy.io.loadmat(load_file_path)
        latent_z = matfile["latent_z"]
        text = matfile["text"]
        text_reconst = matfile["text_reconst"]
    elif args.model == "bert":
        load_file_path = Path(SAVE_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_latent.mat")
        latent_dict = scipy.io.loadmat(load_file_path)
        latent_z = latent_dict["latent_z"]
        text = latent_dict["text"]
        text_reconst = np.asarray([""])
    elif args.model == "glove":
        MAIN_DIR_PATH: Final[str] = config_ini.get("Pereira2018", "main_directory")
        latent_z = load_GloVe(MAIN_DIR_PATH, exp_id=args.exp_id)
        text = np.asarray([""])
        text_reconst = np.asarray([""])
    else:
        raise ValueError(f"{args.model} is not defined.")

    if args.debug:
        latent_z = latent_z[:NUM_FOR_DEBUG]

    # Run Cross-Validation
    for cv, index_list_dict in tqdm(index_dict.items()):
        train_index = index_list_dict["train_index"]
        test_index = index_list_dict["test_index"]
        if args.debug:
            train_index = [i for i in train_index if i < NUM_FOR_DEBUG]
            test_index = [i for i in test_index if i < NUM_FOR_DEBUG]
            print(len(train_index), len(test_index))
        if args.permute:  # For permutation test.
            np.random.seed(cv)
            fmri_data = np.random.permutation(fmri_data)
        train_fmri_data, train_latent_z = fmri_data[train_index], latent_z[train_index]
        test_fmri_data, test_latent_z = fmri_data[test_index], latent_z[test_index]

        pred_latent_z, pipeline = predict_latent(
            {
                "X_train": train_fmri_data,
                "Y_train": train_latent_z,
                "X_test": test_fmri_data,
                "Y_test": test_latent_z,
            },
            reshape=args.reshape,
            feature_selection=args.feature_selection,
            regressor=args.regressor,
        )

        coef = np.mean(pipeline["ridgecv"].coef_, axis=0)  # type:ignore
        coef_abs = np.mean(np.abs(pipeline["ridgecv"].coef_), axis=0)  # type:ignore
        if args.model == "optimus":
            pred_text_reconst = run_optimus_reconst(
                pred_latent_z=pred_latent_z,
                test_text=text[test_index],
                args=args,
            )
            results = {
                "train_latent_z": train_latent_z,
                "test_latent_z": test_latent_z,
                "pred_latent_z": pred_latent_z,
                "train_text": text[train_index],
                "test_text": text[test_index],
                "train_text_reconst": text_reconst[train_index],
                "test_text_reconst": text_reconst[test_index],
                "pred_text_reconst": pred_text_reconst,
                "coef": coef,
                "coef_abs": coef_abs,
            }
        elif args.model == "bert":
            results = {
                "train_latent_z": train_latent_z,
                "test_latent_z": test_latent_z,
                "pred_latent_z": pred_latent_z,
                "train_text": text[train_index],
                "test_text": text[test_index],
                "coef": coef,
                "coef_abs": coef_abs,
            }
        elif args.model == "glove":
            results = {
                "train_latent_z": train_latent_z,
                "test_latent_z": test_latent_z,
                "pred_latent_z": pred_latent_z,
                "coef": coef,
                "coef_abs": coef_abs,
            }
        else:
            raise ValueError(f"{args.model} is not defined.")

        prams = "_".join([args.feature_selection, args.regressor])
        save_file_path = Path(SAVE_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_{prams}_cv{cv}_results.mat")
        if args.permute:
            PERM_SAVE_DIR_PATH: Final[str] = f"./results/{args.model}/permute/exp{args.exp_id}/s{args.sub_id:02}"
            os.makedirs(PERM_SAVE_DIR_PATH, exist_ok=True)
            save_file_path = Path(PERM_SAVE_DIR_PATH).joinpath(
                f"exp{args.exp_id}_s{args.sub_id:02}_{prams}_cv{cv}_results.mat"
            )
        scipy.io.savemat(save_file_path, results)
        if args.debug:
            break


if __name__ == "__main__":
    main()
