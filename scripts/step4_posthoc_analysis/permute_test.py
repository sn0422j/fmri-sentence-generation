from typing import Tuple

import numpy as np
import pandas as pd
from cliffsDelta import cliffsDelta
from scipy import stats


def statistic_test(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float, float, str]:
    if len(data1) != len(data2):
        assert ValueError("length {len(data1)} and {len(data2)} should be the same.")
    statistic, pval = stats.ranksums(x=data1, y=data2)
    delta, size = cliffsDelta(data1, data2)
    return statistic, pval, delta, size


def main():
    stats_df = pd.DataFrame(
        columns=[
            "model",
            "exp_id",
            "sub_id",
            "metrics",
            "mean",
            "std",
            "mean_permute",
            "std_permute",
            "statistic",
            "delta",
            "effect",
            "p-value",
            "significance",
        ]
    )
    for model, model_name in [
        ["glove", "GloVe"],
        ["bert", "BERT"],
        ["optimus", "Optimus"],
    ]:
        result_df = pd.read_csv(f"./results/{model}/results_pereira2018_evaluate.csv")
        result_df = (
            result_df.merge(pd.read_csv(f"./results/{model}/results_pereira2018_evaluate_topic.csv"))
            .sort_values("cv")
            .reset_index(drop=True)
        )
        permute_df = pd.read_csv(f"./results/{model}/results_pereira2018_evaluate_permute.csv")
        permute_df = (
            permute_df.merge(pd.read_csv(f"./results/{model}/results_pereira2018_evaluate_topic_permute.csv"))
            .sort_values("cv")
            .reset_index(drop=True)
        )
        metrics_list = ["Sum match", "Strict match", "Topic Accuracy"]
        if model == "optimus":
            metrics_list = metrics_list + ["BLEU (pred)"] + ["BERTScore (pred)"]
            result_df = (
                result_df.merge(pd.read_csv(f"./results/{model}/results_pereira2018_evaluate_bertscore.csv"))
                .sort_values("cv")
                .reset_index(drop=True)
            )
            permute_df = (
                permute_df.merge(pd.read_csv(f"./results/{model}/results_pereira2018_evaluate_bertscore_permute.csv"))
                .sort_values("cv")
                .reset_index(drop=True)
            )
        for exp_id in [2, 3]:
            exp_result_df = result_df[result_df["exp_id"] == exp_id].copy()
            exp_permute_df = permute_df[permute_df["exp_id"] == exp_id].copy()
            if exp_id == 2:
                sub_id_list = [0, 2, 4, 7, 8, 9, 14, 15]
            elif exp_id == 3:
                sub_id_list = [0, 2, 3, 4, 7, 15]
            else:
                raise ValueError(f"exp {exp_id} not found.")
            for sub_id in sub_id_list:
                for metrics in metrics_list:
                    mean = np.mean(exp_result_df.loc[result_df["sub_id"] == sub_id, metrics].to_numpy())
                    std = np.std(exp_result_df.loc[result_df["sub_id"] == sub_id, metrics].to_numpy())
                    mean_permute = np.mean(exp_permute_df.loc[result_df["sub_id"] == sub_id, metrics].to_numpy())
                    std_permute = np.std(exp_permute_df.loc[result_df["sub_id"] == sub_id, metrics].to_numpy())
                    statistic, p, delta, size = statistic_test(
                        data1=exp_result_df.loc[result_df["sub_id"] == sub_id, metrics].to_numpy(),
                        data2=exp_permute_df.loc[exp_permute_df["sub_id"] == sub_id, metrics].to_numpy(),
                    )
                    pd_series = pd.Series(
                        [
                            model_name,
                            exp_id,
                            sub_id,
                            metrics,
                            mean,
                            std,
                            mean_permute,
                            std_permute,
                            statistic,
                            delta,
                            size,
                            p,
                            "",
                        ],
                        index=stats_df.columns,
                        name=f"{model}-{exp_id}-{sub_id}-{metrics}",
                    )
                    stats_df = stats_df.append(pd_series)  # type: ignore

    stats_df = stats_df.reset_index(drop=True)
    assert isinstance(stats_df, pd.DataFrame)

    N = len(stats_df)
    for i in range(N):
        p = stats_df.loc[i, "p-value"]
        assert isinstance(p, float)
        significance = ""
        if p < 0.05 / N:
            significance = "*"
        if p < 0.005 / N:
            significance = "**"
        if p < 0.0005 / N:
            significance = "***"
        stats_df.loc[i, "significance"] = significance
    stats_df.to_csv(f"./results/results_pereira2018_stat.csv", index=False)

    RENAME_DICT = {
        "metrics": "評価指標",
        "model": "モデル",
        "exp_id": "実験",
        "sub_id": "実験参加者",
        "mean": "平均",
        "std": "標準偏差",
        "mean_permute": "平均(帰無分布)",
        "std_permute": "標準偏差(帰無分布)",
        "p-value": "P値",
        "significance": "有意性基準",
        "delta": "Cliff's delta (d)",
        "effect": "効果量基準",
    }
    ROUND_DICT = {
        "mean": 6,
        "std": 6,
        "mean_permute": 6,
        "std_permute": 6,
        "p-value": 64,
        "delta": 6,
    }
    METRICS_ORDER = {
        "Sum match": 0,
        "Strict match": 1,
        "Topic Accuracy": 2,
        "BERTScore (pred)": 3,
        "BLEU (pred)": 4,
    }
    MODEL_ORDER = {
        "GloVe": 0,
        "BERT": 1,
        "Optimus": 2,
    }
    stats_df = stats_df[list(RENAME_DICT.keys())]
    stats_df = stats_df.round(ROUND_DICT)
    stats_df["metrics_order"] = stats_df["metrics"].replace(METRICS_ORDER)
    stats_df["model_order"] = stats_df["model"].replace(MODEL_ORDER)
    stats_df = stats_df.sort_values(["metrics_order", "model_order", "exp_id", "sub_id"]).drop(
        ["metrics_order", "model_order"], axis=1
    )
    stats_df["metrics"] = stats_df["metrics"].replace(
        {
            "BERTScore (pred)": "BERTScore(平均F1スコア)",
            "Strict match": "Strict match Accuracy",
            "Sum match": "Sum match Accuracy",
            "BLEU (pred)": "BLEUスコア",
        }
    )
    stats_df = stats_df.rename(columns=RENAME_DICT)
    stats_df.to_csv(f"./results/results_pereira2018_stat.csv", index=False)
    stats_df.groupby(["評価指標", "モデル", "実験", "実験参加者"], as_index=False, sort=False).agg(lambda x: x).to_csv(
        f"./results/results_pereira2018_stat_share.csv", index=False
    )


if __name__ == "__main__":
    main()
