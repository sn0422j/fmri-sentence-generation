import argparse

import numpy as np
import torch
import torch.autograd
import torch.backends.cudnn
import torch.cuda
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, Dataset
from utils import add_optimus_path, seed_everything

add_optimus_path()

from pytorch_transformers import BertModel, BertTokenizer


class CustomDataset(Dataset):
    def __init__(self, labels, tokenizer, test_flag=False):
        self.labels = labels
        self.tokenizer = tokenizer
        self.test_flag = test_flag

    def __getitem__(self, index):
        sentence = self.labels[index]
        token = self.tokenize_wrapper(sentence)
        return token

    def __len__(self):
        return len(self.labels)

    def tokenize_wrapper(self, text):
        text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        if self.test_flag:
            print(f"{len(tokenized_text)=}")

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if self.test_flag:
            print(f"{indexed_tokens=}")
        return np.array(indexed_tokens)


def latent_code_from_text(model: nn.Module, dataloader: DataLoader, args):
    model.to(args.device)
    model.eval()
    latent = []
    for inputs in dataloader:
        inputs = inputs.to(args.device).long()
        with torch.autograd.grad_mode.set_grad_enabled(False):
            outputs = model(inputs, args.device)
            latent.append(outputs[0].cpu().numpy())
    return np.array(latent)


class WrappedBERT(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.pretrained_bert = BertModel.from_pretrained("bert-base-cased")

    def forward(self, x, device):
        emmbed = self.pretrained_bert(x)[0]
        mean_emmbed = (
            torch.tensor([self._average_emmbed(emmbed_, x_) for emmbed_, x_ in zip(emmbed, x)]).float().to(device)
        )
        return mean_emmbed

    def _average_emmbed(self, emmbed, x):
        averaged_emmbed = np.zeros(768)
        sentence_length = 0

        for i, x_i in enumerate(x):
            if x_i == 101:
                continue  # [CLS]
            if x_i == 102:
                continue  # [SEP]
            if x_i == 0:
                continue  # [PAD]
            averaged_emmbed += emmbed[i].cpu().detach().numpy()
            sentence_length += 1

        return averaged_emmbed / sentence_length


def prepare_bert(args):
    seed_everything()
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    args.device = device
    args.latent_size = 768

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
    model_bert = WrappedBERT(args.latent_size)
    model_bert.to(device)
    return model_bert, tokenizer, args


def run_test_case():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print(args)

    model_bert, tokenizer, args = prepare_bert(args)
    text_dataset = CustomDataset(["input text 1 .", "input text 2 ."], tokenizer, test_flag=True)
    text_dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False)

    latent = latent_code_from_text(model_bert, text_dataloader, args)
    print(f"latent: {latent.shape}")


if __name__ == "__main__":
    run_test_case()
