import argparse
import glob
import re
from collections import OrderedDict

import pandas as pd
from my_metrics import (
    compute_blue_pred,
    compute_blue_true,
    compute_cosine,
    compute_metrics,
    compute_pearson,
    compute_r2,
    compute_sblue_pred,
    compute_sblue_true,
    compute_strict_match,
    compute_sum_match,
)
from tqdm import tqdm
from visualize import visualize_model_index


def evaluate_and_aggregate(metrics: OrderedDict, model: str) -> None:
    path_list = glob.glob(f"./results/{model}/exp*/s*/exp*_s*_*_cv*_results.mat")
    result_df = pd.DataFrame(columns=["exp_id", "sub_id", "condition", "cv"] + list(metrics.keys()))
    for path in tqdm(path_list):
        exp_id = re.findall(f"{model}/exp(.*)/s", path)[0]
        sub_id = re.findall(f"{model}/exp{exp_id}/s(.*)/exp", path)[0]
        condition = re.findall(f"{model}/exp{exp_id}/s{sub_id}/exp{exp_id}_s{sub_id}_(.*)_cv", path)[0]
        cv = re.findall(f"{model}/exp{exp_id}/s{sub_id}/exp{exp_id}_s{sub_id}_{condition}_cv(.*)_results.mat", path)[0]
        infomation = OrderedDict(
            [
                ("exp_id", str(exp_id)),
                ("sub_id", str(sub_id)),
                ("condition", condition),
                ("cv", str(cv)),
            ]
        )
        pd_series = compute_metrics(path, metrics, infomation)
        result_df = pd.concat([result_df, pd.DataFrame(pd_series).T], axis=0)

    print(result_df.head())
    result_df.to_csv(f"./results/{model}/results_pereira2018_evaluate.csv", index=False)

    path_list = glob.glob(f"./results/{model}/permute/exp*/s*/exp*_s*_*_cv*_results.mat")
    result_df = pd.DataFrame(columns=["exp_id", "sub_id", "condition", "cv"] + list(metrics.keys()))
    for path in tqdm(path_list):
        exp_id = re.findall(f"{model}/permute/exp(.*)/s", path)[0]
        sub_id = re.findall(f"{model}/permute/exp{exp_id}/s(.*)/exp", path)[0]
        condition = re.findall(f"{model}/permute/exp{exp_id}/s{sub_id}/exp{exp_id}_s{sub_id}_(.*)_cv", path)[0]
        cv = re.findall(
            f"{model}/permute/exp{exp_id}/s{sub_id}/exp{exp_id}_s{sub_id}_{condition}_cv(.*)_results.mat", path
        )[0]
        infomation = OrderedDict(
            [
                ("exp_id", str(exp_id)),
                ("sub_id", str(sub_id)),
                ("condition", condition),
                ("cv", str(cv)),
            ]
        )
        pd_series = compute_metrics(path, metrics, infomation)
        result_df = pd.concat([result_df, pd.DataFrame(pd_series).T], axis=0)

    print(result_df.head())
    result_df.to_csv(f"./results/{model}/results_pereira2018_evaluate_permute.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="used model name", type=str)
    args = parser.parse_args()
    print(args)

    if args.model in ["glove", "bert"]:
        metrics = OrderedDict(
            [  # type: ignore
                ("R2 score", compute_r2),
                ("Pearson R", compute_pearson),
                ("Cosine", compute_cosine),
                ("Sum match", compute_sum_match),
                ("Strict match", compute_strict_match),
            ]
        )
    elif args.model in ["optimus"]:
        metrics = OrderedDict(
            [  # type: ignore
                ("R2 score", compute_r2),
                ("Pearson R", compute_pearson),
                ("Cosine", compute_cosine),
                ("Sum match", compute_sum_match),
                ("Strict match", compute_strict_match),
                ("BLEU (true)", compute_blue_true),
                ("BLEU (pred)", compute_blue_pred),
                ("self-BLEU (true)", compute_sblue_true),
                ("self-BLEU (pred)", compute_sblue_pred),
            ]
        )
    else:
        raise ValueError(f"{args.model} is not found.")

    evaluate_and_aggregate(
        metrics=metrics,
        model=args.model,
    )
    visualize_model_index(
        metrics_name_list=["Cosine", "Sum match", "Strict match"],
        model=args.model,
    )


if __name__ == "__main__":
    main()
