import argparse
import copy
import pickle
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm
from utils import check_sub_id
from visualize import plot_barchart


@dataclass
class PretrainedClf:
    clf_list: list = field(default_factory=list)
    clf_mean_acc: float = 0.0
    clf_mean_top5_acc: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, n_splits=6):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        param_grid = [
            {"C": [1, 10, 100], "kernel": ["linear"]},
            {"C": [1, 10, 100], "kernel": ["rbf"], "gamma": [0.1, 0.01, 0.001, 0.0001]},
            {"C": [1, 10, 100], "kernel": ["poly"], "degree": [2, 3, 4], "gamma": [0.1, 0.01, 0.001, 0.0001]},
            {"C": [1, 10, 100], "kernel": ["sigmoid"], "gamma": [0.1, 0.01, 0.001, 0.0001]},
        ]

        acc_score_list = []
        top5_acc_score_list = []
        for train, valid in skf.split(X, y):
            clf = GridSearchCV(
                SVC(random_state=0, max_iter=5000),
                param_grid,
                cv=StratifiedKFold(n_splits=n_splits - 1, shuffle=False),
                n_jobs=-1,
                verbose=1,
            ).fit(X[train], y[train])
            acc_score = accuracy_score(y[valid], clf.predict(X[valid]))
            top5_acc_score = top_k_accuracy_score(
                y[valid], clf.decision_function(X[valid]), k=5, labels=np.arange(1, 25)
            )
            acc_score_list.append(acc_score)
            top5_acc_score_list.append(top5_acc_score)
            print(f"ACC score: {acc_score:.3f}")
            print(f"top-5 ACC score: {top5_acc_score:.3f}")
            self.clf_list.append(copy.copy(clf))
        self.clf_mean_acc = np.mean(acc_score_list).item()
        self.clf_mean_top5_acc = np.mean(top5_acc_score_list).item()
        print(f"Mean of ACC score: {self.clf_mean_acc:.3f}")
        print(f"Mean of top-5 ACC score: {self.clf_mean_top5_acc:.3f}")
        return self

    def predict_test(self, X_true: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
        y_pred_list = []
        y_pred_decision_function_list = []
        for clf in self.clf_list:
            y_pred_list.append(clf.predict(X_true))
            y_pred_decision_function_list.append(clf.decision_function(X_true))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            y_pred = mode(y_pred_list)[0][0]
            # print(f"{y_true=}, {y_pred=}")
        y_pred_decision_function = np.mean(y_pred_decision_function_list, axis=0)
        acc_score = accuracy_score(y_true, y_pred)
        top5_acc_score = top_k_accuracy_score(y_true, y_pred_decision_function, k=5, labels=np.arange(1, 25))
        return float(acc_score), top5_acc_score


def prepare_clf(args: argparse.Namespace) -> PretrainedClf:
    cv = "0"
    path = f"./results/{args.model}/exp{args.exp_id}/s{args.sub_id:02}/exp{args.exp_id}_s{args.sub_id:02}_{args.condition}_cv{cv}_results.mat"
    matfile = scipy.io.loadmat(path)

    load_file_path = f"data/exp{args.exp_id}/s{args.sub_id:02}/exp{args.exp_id}_s{args.sub_id:02}_index.pickle"
    with open(load_file_path, "rb") as f:
        index_dict = pickle.load(f)

    X = np.vstack([matfile["train_latent_z"], matfile["test_latent_z"]])
    y = np.hstack([index_dict[int(cv)]["train_passage"], index_dict[int(cv)]["test_passage"]])
    print(np.unique(y, return_counts=True))

    clf = PretrainedClf().fit(X, y)
    return clf


def compute_topic_acc(
    clf: PretrainedClf,
    path: str,
    index_dict: dict,
    infomation: OrderedDict,
) -> pd.Series:
    index = list(infomation.keys()) + ["Topic Accuracy", "Topic top-5 Accuracy"]
    pd_series = pd.Series(np.zeros(len(index)), index=index, name=f"_".join(list(infomation.keys())))
    matfile = scipy.io.loadmat(path)

    for info_name, info_value in infomation.items():
        pd_series[info_name] = info_value

    X = matfile["pred_latent_z"]
    y = np.array(index_dict[int(infomation["cv"])]["test_passage"])
    pd_series[["Topic Accuracy", "Topic top-5 Accuracy"]] = clf.predict_test(X, y)
    pd_series["Topic Accuracy (train)"] = clf.clf_mean_acc
    pd_series["Topic top-5 Accuracy (train)"] = clf.clf_mean_top5_acc

    return pd_series.copy()


def aggregate_accuracy(
    result_df: pd.DataFrame,
    args: argparse.Namespace,
    cv_num=100,
) -> pd.DataFrame:
    for sub_id in tqdm(args.sub_id_list):
        args.sub_id = sub_id
        check_sub_id(exp_id=args.exp_id, sub_id=args.sub_id)
        try:
            clf  # type: ignore
        except:
            clf = prepare_clf(args)

        load_file_path = f"data/exp{args.exp_id}/s{args.sub_id:02}/exp{args.exp_id}_s{args.sub_id:02}_index.pickle"
        with open(load_file_path, "rb") as f:
            index_dict = pickle.load(f)
        for cv in range(cv_num):
            path = f"./results/{args.model}/exp{args.exp_id}/s{args.sub_id:02}/exp{args.exp_id}_s{args.sub_id:02}_{args.condition}_cv{cv}_results.mat"
            if args.permute:
                path = f"./results/{args.model}/permute/exp{args.exp_id}/s{args.sub_id:02}/exp{args.exp_id}_s{args.sub_id:02}_{args.condition}_cv{cv}_results.mat"
            infomation = OrderedDict(
                [
                    ("exp_id", str(args.exp_id)),
                    ("sub_id", str(args.sub_id)),
                    ("condition", args.condition),
                    ("cv", str(cv)),
                ]
            )
            pd_series = compute_topic_acc(
                clf=clf,  # type: ignore
                path=path,
                index_dict=index_dict,
                infomation=infomation,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                result_df = result_df.append(pd_series)  # type: ignore
    return result_df.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="used model name", type=str)
    parser.add_argument("--condition", default="none_ridge")
    parser.add_argument("--debug", help="(option) flag of debug", action="store_true")
    parser.add_argument("--permute", help="(option) flag of permute test", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.model in ["glove", "bert", "optimus"]:
        pass
    else:
        raise ValueError(f"{args.model} is not found.")

    result_df = pd.DataFrame(
        columns=["exp_id", "sub_id", "condition", "cv"]
        + ["Topic Accuracy", "Topic Accuracy (train)"]
        + ["Topic top-5 Accuracy", "Topic top-5 Accuracy (train)"]
    )
    args.exp_id = 2
    args.sub_id_list = [0, 2, 4, 7, 8, 9, 14, 15]
    result_df = aggregate_accuracy(result_df, args)
    args.exp_id = 3
    args.sub_id_list = [0, 2, 3, 4, 7, 15]
    result_df = aggregate_accuracy(result_df, args)

    print(result_df.head())
    if args.permute:
        result_df.to_csv(f"./results/{args.model}/results_pereira2018_evaluate_topic_permute.csv", index=False)
    else:
        result_df.to_csv(f"./results/{args.model}/results_pereira2018_evaluate_topic.csv", index=False)
        result_df = result_df.astype({"exp_id": int, "sub_id": int})
        plot_barchart(result_df=result_df.copy(), model=args.model, metrics_name="Topic Accuracy", ylim=(0, 1))
        plot_barchart(result_df=result_df.copy(), model=args.model, metrics_name="Topic top-5 Accuracy", ylim=(0, 1))


if __name__ == "__main__":
    main()
