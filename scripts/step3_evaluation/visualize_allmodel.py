from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]


def float_rgb(rgb):
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


COLOR_DICT: Final[dict] = {
    "red": float_rgb((255, 75, 0)),
    "yellow": float_rgb((255, 241, 0)),
    "green": float_rgb((3, 175, 122)),
    "blue": float_rgb((0, 90, 255)),
    "skyblue": float_rgb((77, 196, 255)),
    "pink": float_rgb((255, 128, 130)),
    "orange": float_rgb((246, 170, 0)),
    "purple": float_rgb((201, 172, 230)),
    "blown": float_rgb((128, 64, 25)),
    "black": float_rgb((0, 0, 0)),
    "white": float_rgb((255, 255, 255)),
    "grey": float_rgb((132, 145, 158)),
    "lightgrey": float_rgb((200, 200, 203)),
}


def plot_barchart_allmodel(result_df: pd.DataFrame, metrics_name: str, ylim: tuple):
    # sns.set_context("paper", 2)
    sns.set_theme("paper", font="Noto Sans CJK JP", font_scale=2)
    plt.style.use("ggplot")
    f, ax = plt.subplots(1, 1, figsize=(13, 6), dpi=150)
    palette = sns.color_palette([COLOR_DICT["blue"], COLOR_DICT["red"], COLOR_DICT["green"]])
    sns.barplot(x="exp_id", y=metrics_name, hue="model", errwidth=0, palette=palette, data=result_df, ax=ax)
    sns.stripplot(
        x="exp_id", y=metrics_name, hue="model", jitter=True, dodge=True, color="black", size=6, data=result_df, ax=ax
    )
    ax.set_xlabel("実験", fontsize=20)
    ax.set_ylabel(metrics_name, fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    labels_length = len(handles)
    ax.legend(
        handles[labels_length // 2 : labels_length],
        labels[labels_length // 2 : labels_length],
        frameon=False,
        title="Model",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )
    ax.set_xticklabels(["2", "3"])
    ax.set_ylim(ylim[0], ylim[1])
    metrics_name = metrics_name.replace(" ", "_")
    plt.savefig(f"./results/pereira2018_{metrics_name}_allmodel_jp.png", bbox_inches="tight")
    plt.savefig(f"./results/pereira2018_{metrics_name}_allmodel_jp.svg", bbox_inches="tight")


def main():
    result_df_list = []

    for model, model_name in [
        ["glove", "GloVe"],
        ["bert", "BERT"],
        ["optimus", "Optimus"],
    ]:
        result_df = pd.read_csv(f"./results/{model}/results_pereira2018_evaluate.csv")
        result_df = result_df.merge(pd.read_csv(f"./results/{model}/results_pereira2018_evaluate_topic.csv"))
        result_df = result_df.astype({"exp_id": int, "sub_id": int})
        if model == "optimus":
            result_df = result_df.merge(pd.read_csv(f"./results/{model}/results_pereira2018_evaluate_bertscore.csv"))
            result_df = result_df.astype({"exp_id": int, "sub_id": int})
            result_df["BLEU (relative)"] = result_df["BLEU (pred)"] / result_df["BLEU (true)"]
            result_df["self-BLEU (relative)"] = result_df["self-BLEU (pred)"] / result_df["self-BLEU (true)"]
        result_df = result_df.groupby(["sub_id", "exp_id"], as_index=False).mean().drop(columns="cv")
        result_df = result_df.assign(model=model_name)
        result_df_list.append(result_df)

    result_df = pd.concat(result_df_list)
    result_df.groupby(["exp_id", "model"], as_index=False).mean().to_csv(
        "./results/results_pereira2018_evaluate.csv", index=False
    )
    plot_barchart_allmodel(result_df, "Sum match", (0, 1))
    plot_barchart_allmodel(result_df, "Strict match", (0, 1))
    plot_barchart_allmodel(result_df, "Topic Accuracy", (0, 0.2))
    plot_barchart_allmodel(result_df, "BLEU (pred)", (0, 0.25))
    plot_barchart_allmodel(result_df, "BERTScore (pred)", (0, 1))


if __name__ == "__main__":
    main()
