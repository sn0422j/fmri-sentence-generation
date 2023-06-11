from collections import OrderedDict
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import scipy.io
import torch
from bert_score import score
from metrics_blue import Bleu, SelfBleu
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity


def compute_r2(matfile):
    y_true = matfile["test_latent_z"]
    y_pred = matfile["pred_latent_z"]
    return r2_score(y_true=y_true, y_pred=y_pred)


def compute_pearson(matfile):
    y_true = matfile["test_latent_z"]
    y_pred = matfile["pred_latent_z"]
    return np.corrcoef(y_true, y_pred)


def compute_cosine(matfile):
    y_true = matfile["test_latent_z"]
    y_pred = matfile["pred_latent_z"]
    return cosine_similarity(y_true, y_pred)


def compute_sum_match(matfile):
    """
    Beinborn, L., Abnar, S., & Choenni, R. (2019).
    Robust evaluation of language-brain encoding experiments.
    arXiv preprint arXiv:1904.02547.
    """
    y_true = matfile["test_latent_z"]
    y_pred = matfile["pred_latent_z"]

    ac_list = []
    cos_sim = cosine_similarity(y_pred, y_true)
    for index1, index2 in combinations(list(range(len(y_pred))), 2):
        same = cos_sim[index1, index1] + cos_sim[index2, index2]
        diff = cos_sim[index1, index2] + cos_sim[index2, index1]
        if same - diff > 0:
            ac_list.append(1)
        elif same - diff == 0:
            print("zero")
            ac_list.append(0)
        else:
            ac_list.append(0)

    return np.array(ac_list)


def compute_strict_match(matfile):
    """
    Beinborn, L., Abnar, S., & Choenni, R. (2019).
    Robust evaluation of language-brain encoding experiments.
    arXiv preprint arXiv:1904.02547.
    """
    y_true = matfile["test_latent_z"]
    y_pred = matfile["pred_latent_z"]

    ac_list = []
    cos_sim = cosine_similarity(y_pred, y_true)
    for index1, index2 in combinations(list(range(len(y_pred))), 2):
        single1 = 1 if cos_sim[index1, index1] > cos_sim[index1, index2] else 0
        single2 = 1 if cos_sim[index2, index2] > cos_sim[index2, index1] else 0
        if single1 + single2 == 2:
            ac_list.append(1)
        else:
            ac_list.append(0)

    return np.array(ac_list)


def compute_blue(text_true: List[str], text_pred: List[str]):
    return Bleu(
        test_text=text_pred,  # type: ignore
        real_text=text_true,  # type: ignore
        num_real_sentences=len(text_pred),
        num_fake_sentences=len(text_true),
        gram=5,
    ).get_score()


def compute_blue_true(matfile) -> float:
    text_true = matfile["test_text"]
    text_pred = matfile["test_text_reconst"]
    for i in range(len(text_pred)):
        text_pred[i] = text_pred[i].replace(".", " .")

    bleu5 = compute_blue(text_true, text_pred)
    return bleu5


def compute_blue_pred(matfile) -> float:
    text_true = matfile["test_text"]
    text_pred = matfile["pred_text_reconst"]
    for i in range(len(text_pred)):
        text_pred[i] = text_pred[i].replace(".", " .")

    bleu5 = compute_blue(text_true, text_pred)
    return bleu5


def compute_sblue_true(matfile) -> float:
    text_pred = matfile["test_text_reconst"]
    for i in range(len(text_pred)):
        text_pred[i] = text_pred[i].replace(".", " .")

    sbleu5 = SelfBleu(test_text=text_pred, num_sentences=len(text_pred), gram=5).get_score()
    return sbleu5


def compute_sblue_pred(matfile) -> float:
    text_pred = matfile["pred_text_reconst"]
    for i in range(len(text_pred)):
        text_pred[i] = text_pred[i].replace(".", " .")

    sbleu5 = SelfBleu(test_text=text_pred, num_sentences=len(text_pred), gram=5).get_score()
    return sbleu5


def compute_bert_score(text_true: List[str], text_pred: List[str]):
    _, _, f1 = score(text_pred, text_true, lang="en", device="cuda")
    return torch.mean(f1).item()  # type: ignore


def compute_bert_score_true(matfile) -> float:
    text_true = matfile["test_text"]
    text_pred = matfile["test_text_reconst"]
    for i in range(len(text_pred)):
        text_pred[i] = text_pred[i].replace(".", " .")
    assert isinstance(text_true, np.ndarray)
    assert isinstance(text_pred, np.ndarray)
    text_true = text_true.flatten().tolist()
    text_pred = text_pred.flatten().tolist()

    return compute_bert_score(text_true, text_pred)


def compute_bert_score_pred(matfile) -> float:
    text_true = matfile["test_text"]
    text_pred = matfile["pred_text_reconst"]
    for i in range(len(text_pred)):
        text_pred[i] = text_pred[i].replace(".", " .")
    assert isinstance(text_true, np.ndarray)
    assert isinstance(text_pred, np.ndarray)
    text_true = text_true.flatten().tolist()
    text_pred = text_pred.flatten().tolist()

    return compute_bert_score(text_true, text_pred)


def compute_metrics(path: str, metrics: OrderedDict, infomation: OrderedDict) -> pd.Series:
    """
    metrics: OrderedDict
        - (*metrics name): function
    infomation: OrderedDict
        - exp_id: str
        - sub_id: str
        - condition: str
        - cv: str
    """
    index = list(infomation.keys()) + list(metrics.keys())
    pd_series = pd.Series(np.zeros(len(index)), index=index, name=f"_".join(list(infomation.keys())))
    matfile = scipy.io.loadmat(path)

    for info_name, info_value in infomation.items():
        pd_series[info_name] = info_value

    for metrics_name, metrics_function in metrics.items():
        pd_series[metrics_name] = np.mean(metrics_function(matfile))

    return pd_series
