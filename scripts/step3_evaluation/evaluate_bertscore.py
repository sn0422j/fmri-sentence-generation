import argparse
import glob
import re
from collections import OrderedDict

import pandas as pd
from my_metrics import compute_bert_score_pred, compute_bert_score_true, compute_metrics
from tqdm import tqdm


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
        result_df.to_csv(f"./results/{model}/results_pereira2018_evaluate_bertscore.csv", index=False)

    print(result_df.head())

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
        result_df.to_csv(f"./results/{model}/results_pereira2018_evaluate_bertscore_permute.csv", index=False)

    print(result_df.head())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="used model name", type=str)
    args = parser.parse_args()
    print(args)

    if args.model in ["optimus"]:
        metrics = OrderedDict(
            [
                ("BERTScore (true)", compute_bert_score_true),
                ("BERTScore (pred)", compute_bert_score_pred),
            ]
        )
    else:
        raise ValueError(f"{args.model} is not found.")

    evaluate_and_aggregate(metrics=metrics, model=args.model)


if __name__ == "__main__":
    main()
