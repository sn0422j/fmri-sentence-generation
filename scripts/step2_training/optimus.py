import argparse
import os
from configparser import ConfigParser

import numpy as np
import torch
import torch.autograd
import torch.backends.cudnn
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from utils import add_optimus_path

add_optimus_path()


from modules import VAE
from pytorch_transformers import (
    BertConfig,
    BertForLatentConnector,
    BertTokenizer,
    GPT2Config,
    GPT2ForLatentConnector,
    GPT2Tokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    "bert": (BertConfig, BertForLatentConnector, BertTokenizer),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_step = 31250


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_vae(args):
    set_seed(1234)

    config_ini = ConfigParser()
    config_ini.read("config.ini", encoding="utf-8")

    # set the checkpoint filr of the VAE
    output_encoder_dir = config_ini.get("VAE", "output_encoder_dir")
    output_decoder_dir = config_ini.get("VAE", "output_decoder_dir")
    checkpoint_dir = config_ini.get("VAE", "checkpoint_dir")

    # add args
    args.temperature = 1.0
    args.top_k = 0
    args.top_p = 1.0
    args.device = device
    args.block_size = 100
    args.latent_size = 768
    print(args)

    # load a trained Encoder model and vocabulary
    _, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES["bert"]
    model_encoder = encoder_model_class.from_pretrained(output_encoder_dir, latent_size=args.latent_size)
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained("bert-base-cased", do_lower_case=True)
    model_encoder.to(device)
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    # load a trained Decoder model and vocabulary
    _, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES["gpt2"]
    model_decoder = decoder_model_class.from_pretrained(output_decoder_dir, latent_size=args.latent_size)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained("gpt2", do_lower_case=True)
    model_decoder.to(device)
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

    # add Padding token to GPT2
    special_tokens_dict = {"pad_token": "<PAD>", "bos_token": "<BOS>", "eos_token": "<EOS>"}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens to GPT2")
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    assert tokenizer_decoder.pad_token == "<PAD>"

    # Load full model
    output_full_dir = os.path.join(checkpoint_dir, f"checkpoint-full-{global_step}")
    checkpoint = torch.load(os.path.join(output_full_dir, "training.bin"))
    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
    model_vae.load_state_dict(checkpoint["model_state_dict"])
    model_vae.to(device)

    return model_vae, tokenizer_encoder, tokenizer_decoder, args


def latent_code_from_text(text, tokenizer_encoder, model_vae, args):
    tokenized1 = tokenizer_encoder.encode(text)
    tokenized1 = [101] + tokenized1 + [102]
    coded1 = torch.Tensor([tokenized1])
    coded1 = torch.Tensor.long(coded1)
    with torch.no_grad():
        x0 = coded1
        x0 = x0.to(args.device)
        pooled_hidden_fea = model_vae.encoder(x0, attention_mask=(x0 > 0).float())[1]
        mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        latent_z = mean.squeeze(1)
        coded_length = len(tokenized1)
        return latent_z, coded_length


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence_conditional(
    model,
    length,
    context,
    past=None,
    num_samples=1,
    temperature=1,
    top_k=0,
    top_p=0.0,
    device="cpu",
    decoder_tokenizer=None,
):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        while True:
            # for _ in trange(length):
            inputs = {"input_ids": generated, "past": past}
            outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            # pdb.set_trace()
            if next_token.unsqueeze(0)[0, 0].item() == decoder_tokenizer.encode("<EOS>")[0]:  # type: ignore
                break
            if generated.shape[1] > length:
                break

    return generated


def text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder):
    context_tokens = tokenizer_decoder.encode("<BOS>")

    length = 128
    out = sample_sequence_conditional(
        model=model_vae.decoder,
        context=context_tokens,
        past=latent_z,
        length=length,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        decoder_tokenizer=tokenizer_decoder,
    )
    text_x1 = tokenizer_decoder.decode(out[0, :].tolist(), clean_up_tokenization_spaces=True)
    text_x1 = text_x1.split()[1:-1]
    text_x1 = " ".join(text_x1)
    return text_x1


def generate_latent_text(model_vae: nn.Module, tokenizer_encoder, tokenizer_decoder, args, text):
    latent_z, _ = latent_code_from_text(text, tokenizer_encoder, model_vae, args)
    text_reconst = text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder)
    return latent_z, text_reconst


def run_test_case():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_text")
    args = parser.parse_args()
    print(args)

    model_vae, tokenizer_encoder, tokenizer_decoder, args = prepare_vae(args)
    latent_z, _ = generate_latent_text(model_vae, tokenizer_encoder, tokenizer_decoder, args, args.input_text)
    print(f"latent_z: {latent_z.shape}")

    text_reconst = text_from_latent_code(latent_z, model_vae, args, tokenizer_decoder)
    print(f"text_reconst: {text_reconst}")


if __name__ == "__main__":
    run_test_case()
