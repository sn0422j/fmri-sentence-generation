import pickle

import pandas as pd
import scipy.io


def main():
    text_file = f"./results/optimus/results_pereira2018_best_reconst_.txt"
    with open(text_file, mode="w") as f:
        f.write("from search_best_reconst.py\n")

    result_df = pd.read_csv("./results/optimus/results_pereira2018_evaluate_topic.csv")
    result_df = result_df.sort_values("Topic Accuracy", ascending=False).reset_index(drop=True)
    result_df = (
        result_df.merge(pd.read_csv(f"./results/optimus/results_pereira2018_evaluate_bertscore.csv"))
        .sort_values("cv")
        .reset_index(drop=True)
    )
    top_k = 1
    for exp_id in [2, 3]:
        for i, row in result_df[result_df["exp_id"] == 2].reset_index(drop=True).iterrows():
            assert isinstance(i, int)
            if i == top_k:
                break
            sub_id = str(int(row["sub_id"])).zfill(2)
            condition = row["condition"]
            cv = int(row["cv"])
            path = f"./results/optimus/exp{exp_id}/s{sub_id}/exp{exp_id}_s{sub_id}_{condition}_cv{cv}_results.mat"
            result_matfile = scipy.io.loadmat(path)
            path = f"data/exp{exp_id}/s{sub_id}/exp{exp_id}_s{sub_id}_index.pickle"
            with open(path, "rb") as f:
                index_dict = pickle.load(f)
            path = f"./data/exp{exp_id}/s{sub_id}/exp{exp_id}_s{sub_id}_label.mat"
            label_matfile = scipy.io.loadmat(path)
            category_name = label_matfile["category_name"][0]
            with open(text_file, mode="a") as f:
                f.write(
                    f"\n[Topic Accuracy top{i+1} in exp{exp_id}] [s{sub_id}/exp{exp_id}_s{sub_id}_{condition}_cv{cv}_results]\n"
                )
                f.flush()
                for category, text1, text2 in zip(
                    index_dict[cv]["test_passage"], result_matfile["test_text"], result_matfile["pred_text_reconst"]
                ):
                    category = category_name[category - 1][0]
                    f.write(f"{category}: {text1}\n   -> {text2}\n")
                    f.flush()


if __name__ == "__main__":
    main()
