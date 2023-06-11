import argparse
import os
from pathlib import Path
from typing import Final

import numpy as np
import scipy.io
from tqdm import tqdm
from utils import check_sub_id


def run_bert_generate(labels, args):
    from bert import CustomDataset, latent_code_from_text, prepare_bert
    from torch.utils.data.dataloader import DataLoader

    def run_latent_code_from_text(labels, model_bert, tokenizer, args):
        caption_list = []
        for label in tqdm(labels):
            caption = label[1][0].lower().replace(".", " .")  # He is a student. -> he is a student .
            caption_list.append(caption)
        caption_list = np.array(caption_list)

        text_dataset = CustomDataset(caption_list, tokenizer)
        text_dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False)
        latent = latent_code_from_text(model_bert, text_dataloader, args)
        return latent, caption_list

    model_bert, tokenizer, args = prepare_bert(args)
    latent_z, text = run_latent_code_from_text(labels, model_bert, tokenizer, args)

    results = {
        "latent_z": latent_z,
        "text": text,
    }
    return results



def run_optimus_generate(labels: np.ndarray, args: argparse.Namespace) -> dict:
    from optimus import generate_latent_text, prepare_vae

    def run_generate_latent_text(labels, model_vae, tokenizer_encoder, tokenizer_decoder, args):
        latent_z_list = []
        caption_list = []
        text_reconst_list = []
        for label in tqdm(labels):
            caption = label[1][0].lower().replace(".", " .")  # He is a student. -> he is a student .
            latent_z, text_reconst = generate_latent_text(
                model_vae, tokenizer_encoder, tokenizer_decoder, args, caption
            )
            latent_z_list.append(latent_z.cpu().numpy()[0])
            caption_list.append(caption)
            text_reconst_list.append(text_reconst)
        return np.array(latent_z_list), np.array(caption_list), np.array(text_reconst_list)

    model_vae, tokenizer_encoder, tokenizer_decoder, args = prepare_vae(args)
    latent_z, text, text_reconst = run_generate_latent_text(
        labels, model_vae, tokenizer_encoder, tokenizer_decoder, args
    )
    results = {
        "latent_z": latent_z,
        "text": text,
        "text_reconst": text_reconst,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", help="experiment id from [2,3]", type=int)
    parser.add_argument("sub_id", help="subject id (P01 is sub_id=0)", type=int)
    parser.add_argument("model", help="language model name", type=str)
    parser.add_argument("--debug", help="(option) flag of debug", action="store_true")
    args = parser.parse_args()
    print(args)

    check_sub_id(exp_id=args.exp_id, sub_id=args.sub_id)
    if args.model == "glove":
        exit()

    # Load labels in .mat file
    LOAD_DIR_PATH: Final[str] = f"./data/exp{args.exp_id}/s{args.sub_id:02}"
    load_file_path = Path(LOAD_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_label.mat")
    labels = scipy.io.loadmat(load_file_path)["labels"]
    if args.debug:
        labels = labels[:10]

    # Generate latent z and reconstructed text
    if args.model == "optimus":
        results = run_optimus_generate(labels, args)
    elif args.model == "bert":
        results = run_bert_generate(labels, args)
    else:
        raise ValueError(f"{args.model} is not defined.")

    # Save results in .mat file
    SAVE_DIR_PATH: Final[str] = f"./results/{args.model}/exp{args.exp_id}/s{args.sub_id:02}"
    os.makedirs(SAVE_DIR_PATH, exist_ok=True)
    save_file_name = Path(SAVE_DIR_PATH).joinpath(f"exp{args.exp_id}_s{args.sub_id:02}_latent.mat")
    scipy.io.savemat(save_file_name, results)


if __name__ == "__main__":
    main()
