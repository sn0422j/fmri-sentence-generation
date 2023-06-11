import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_barchart(result_df: pd.DataFrame, model: str, metrics_name: str, ylim: tuple) -> None:
    sns.set_context("paper", 1.6)  # type: ignore
    f, ax = plt.subplots(1, 1, figsize=(13, 6), dpi=150)
    sns.barplot(x="exp_id", y=metrics_name, hue="sub_id", data=result_df, ax=ax)
    sns.stripplot(
        x="exp_id", y=metrics_name, hue="sub_id", jitter=True, dodge=True, color="black", size=2, data=result_df, ax=ax
    )
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metrics_name)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[9 : 9 * 2],
        labels[9 : 9 * 2],
        frameon=False,
        title="Subject",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )
    ax.set_xticklabels(["2", "3"])
    ax.set_ylim(ylim[0], ylim[1])
    plt.tight_layout()
    metrics_name = metrics_name.replace(" ", "_")
    plt.savefig(f"./results/{model}/pereira2018_{metrics_name}.png")


def plot_barchart_permute(result_df: pd.DataFrame, model: str, metrics_name: str, ylim: tuple) -> None:
    sns.set_context("paper", 1.6)  # type: ignore
    f, ax = plt.subplots(1, 1, figsize=(13, 6), dpi=150)
    sns.barplot(x="exp_id", y=metrics_name, hue="sub_id", edgecolor="black", data=result_df, ax=ax)
    sns.stripplot(
        x="exp_id", y=metrics_name, hue="sub_id", jitter=True, dodge=True, color="black", size=2, data=result_df, ax=ax
    )
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metrics_name)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[18 : 18 * 2],
        labels[18 : 18 * 2],
        frameon=False,
        title="Subject",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )
    ax.set_xticklabels(["2", "3"])
    ax.set_ylim(ylim[0], ylim[1])
    plt.tight_layout()
    metrics_name = metrics_name.replace(" ", "_")
    plt.savefig(f"./results/{model}/pereira2018_{metrics_name}_permute.png")


def visualize_model_index(
    metrics_name_list: list,
    model: str,
) -> None:
    result_df = pd.read_csv(f"./results/{model}/results_pereira2018_evaluate.csv")
    result_df = result_df.astype({"exp_id": int, "sub_id": int})
    for metrics_name in metrics_name_list:
        plot_barchart(result_df=result_df.copy(), model=model, metrics_name=metrics_name, ylim=(0, 1))

    permute_df = pd.read_csv(f"./results/{model}/results_pereira2018_evaluate_permute.csv")
    permute_df = permute_df.astype({"exp_id": int, "sub_id": int})
    permute_df["sub_id"] = permute_df["sub_id"].apply(lambda x: str(x).zfill(2) + "-permute")
    result_df["sub_id"] = result_df["sub_id"].apply(lambda x: str(x).zfill(2))
    result_df = pd.concat([result_df, permute_df])
    result_df = result_df.sort_values("sub_id").reset_index(drop=True)
    for metrics_name in metrics_name_list:
        plot_barchart_permute(result_df=result_df.copy(), model=model, metrics_name=metrics_name, ylim=(0, 1))
