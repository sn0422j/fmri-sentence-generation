import os
from configparser import ConfigParser
from typing import Dict, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
from tqdm import tqdm

sns.set_style("whitegrid")
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]


def float_rgb(rgb):
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


COLOR_DICT: Final[dict] = {
    "red": float_rgb((255, 75, 0)),
    "yellow": float_rgb((255, 241, 0)),
    "green": float_rgb((119, 217, 168)),
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


def aal_pallete(label, color_dict):
    label2color_dict = {
        "Temporal_Pole_Sup_L": color_dict["red"],
        "Cerebelum_Crus1_L": color_dict["blown"],
        "Frontal_Sup_Medial_L": color_dict["green"],
        "Frontal_Sup_L": color_dict["blue"],
        "Frontal_Sup_R": color_dict["skyblue"],
        "ParaHippocampal_L": color_dict["pink"],
        "Temporal_Inf_L": color_dict["grey"],
        "Temporal_Pole_Sup_R": color_dict["orange"],
        "Temporal_Inf_R": color_dict["lightgrey"],
        "Insula_R": color_dict["purple"],
        "Temporal_Mid_L": color_dict["yellow"],
    }
    return sns.color_palette([label2color_dict[lab] for lab in label])


def aal_hatches(label):
    label2color_dict = {
        "Temporal_Pole_Sup_L": "/" * 2,
        "Cerebelum_Crus1_L": "\\" * 2,
        "Frontal_Sup_Medial_L": "|" * 2,
        "Frontal_Sup_L": "-" * 2,
        "Frontal_Sup_R": "/" * 2,
        "ParaHippocampal_L": "\\" * 2,
        "Temporal_Inf_L": "|" * 2,
        "Temporal_Pole_Sup_R": "-" * 2,
        "Temporal_Inf_R": "x" * 2,
        "Insula_R": "/" * 2,
        "Temporal_Mid_L": "\\" * 2,
    }
    return [label2color_dict[lab] for lab in label]


def load_roi_infomation(matfile_path: str) -> Dict[str, np.ndarray]:
    matfile = scipy.io.loadmat(matfile_path)
    keys = [
        "dimx",
        "dimy",
        "dimz",
        "dimensions",
        "indicesIn3D",
        "colToCoord",
        "coordToCol",
        "voxelsToNeighbours",
        "numberOfNeighbours",
        "roiMultimaskAAL",
        "roiMultimaskGordon",
        "roisAAL",
        "roisGordon",
        "roiColumnsAAL",
        "roiColumnsGordon",
    ]
    meta = {}
    for i, key in enumerate(keys):
        meta[key] = matfile["meta"][0, 0][i]
    return meta


def load_coefficient(model: str, exp_id: str, sub_id: str, condition: str, cv_num: int = 100) -> np.ndarray:
    for cv in range(cv_num):
        path = f"./results/{model}/exp{exp_id}/s{sub_id}/exp{exp_id}_s{sub_id}_{condition}_cv{cv}_results.mat"
        result_matfile = scipy.io.loadmat(path)
        try:
            coefficient = np.vstack([coefficient, result_matfile["coef_abs"]])  # type: ignore
        except:
            coefficient = result_matfile["coef_abs"]
    return np.mean(coefficient, axis=0).flatten()  # type: ignore


def count_voxels(
    coefficient: np.ndarray,
    roi_infomation: Dict[str, np.ndarray],
    atlas_name: str,
    percentile: int = 95,
) -> pd.DataFrame:
    indexs = np.where(coefficient > np.percentile(coefficient, percentile))
    roi_names = roi_infomation[f"rois{atlas_name}"]
    roi_mask = roi_infomation[f"roiMultimask{atlas_name}"]
    roi_indexs, counts = np.unique(roi_mask[indexs], return_counts=True)
    results = {}
    for i, c in zip(roi_indexs, counts):
        if i == 0:
            continue
        else:
            results[roi_names[i - 1, 0][0]] = c
    results_df = (
        pd.DataFrame.from_dict(results, orient="index", columns=["counts"])
        .sort_values("counts", ascending=False)
        .reset_index()
    )
    if atlas_name == "Gordon":
        results_df["network"] = results_df["index"].apply(lambda x: x.split("_")[1])
        results_df = (
            results_df.groupby("network", as_index=False)
            .sum()
            .sort_values("counts", ascending=False)
            .reset_index(drop=True)
        )
    results_df["percent"] = results_df["counts"] / results_df["counts"].sum(axis=None)
    return results_df.copy()


def main():
    config_ini = ConfigParser()
    config_ini.read("config.ini", encoding="utf-8")
    MAIN_DIR_PATH: Final[str] = config_ini.get("Pereira2018", "main_directory")

    for model, model_name in [
        ["glove", "GloVe"],
        ["bert", "BERT"],
        ["optimus", "Optimus"],
    ]:
        for exp_id in [2, 3]:
            if exp_id == 2:
                sub_id_list = [0, 2, 4, 7, 8, 9, 14, 15]
                number_items = 384
            elif exp_id == 3:
                sub_id_list = [0, 2, 3, 4, 7, 15]
                number_items = 243
            else:
                raise ValueError(f"exp {exp_id} not found.")
            for sub_id in tqdm(sub_id_list):
                if sub_id == 0:
                    sub_name = "P01"
                else:
                    sub_name = f"M{sub_id:02}"
                matfile_path = os.path.join(
                    MAIN_DIR_PATH, f"{sub_name}/{sub_name}/examples_{number_items}sentences.mat"
                )
                roi_infomation = load_roi_infomation(matfile_path)
                coefficient = load_coefficient(
                    model=model, exp_id=str(exp_id), sub_id=str(sub_id).zfill(2), condition="none_ridge"
                )
                results_df_aal_temp = count_voxels(coefficient, roi_infomation, atlas_name="AAL")
                results_df_gordon_temp = count_voxels(coefficient, roi_infomation, atlas_name="Gordon")
                results_df_aal_temp = results_df_aal_temp.assign(exp_sub=f"{exp_id}-{sub_id}")
                results_df_gordon_temp = results_df_gordon_temp.assign(exp_sub=f"{exp_id}-{sub_id}")
                try:
                    results_df_aal = results_df_aal.append(results_df_aal_temp) # type: ignore
                    results_df_gordon = results_df_gordon.append(results_df_gordon_temp) # type: ignore
                except:
                    results_df_aal = results_df_aal_temp.copy()
                    results_df_gordon = results_df_gordon_temp.copy()

        assert isinstance(results_df_aal, pd.DataFrame) # type: ignore
        assert isinstance(results_df_gordon, pd.DataFrame) # type: ignore
        results_df_aal = (
            results_df_aal.groupby("index", as_index=False)
            .mean()
            .sort_values("percent", ascending=False)
            .reset_index(drop=True)
        )
        results_df_gordon = (
            results_df_gordon.groupby("network", as_index=False)
            .mean()
            .sort_values("percent", ascending=False)
            .reset_index(drop=True)
        )
        results_df_aal.to_csv(f"./results/{model}/pereira2018_{model}_aal_rois.csv", index=False)
        results_df_gordon.to_csv(f"./results/{model}/pereira2018_{model}_gordon_rois.csv", index=False)

# def main():
#     for model, model_name in [
#         ["glove", "GloVe"],
#         ["bert", "BERT"],
#         ["optimus", "Optimus"],
#     ]:
#         results_df_aal = pd.read_csv(f"./results/{model}/pereira2018_{model}_aal_rois.csv")
#         results_df_gordon = pd.read_csv(f"./results/{model}/pereira2018_{model}_gordon_rois.csv")

        roi_jp_dict = {
            "Temporal_Pole_Sup_L": "左上側頭極",
            "Temporal_Pole_Sup_R": "右上側頭極",
            "Cerebelum_Crus1_L": "左小脳 Crus I",
            "Cerebelum_Crus1_R": "右小脳 Crus I",
            "Frontal_Sup_Medial_L": "左内側上前頭回",  #  (Superior frontal gyrus, medial): 内側上前頭回，
            "Frontal_Sup_Medial_R": "右内側上前頭回",  #  (Superior frontal gyrus, medial): 内側上前頭回，
            "Frontal_Sup_L": "左上前頭回",  #  (Superior frontal gyrus): 上前頭回，
            "Frontal_Sup_R": "右上前頭回",  #  (Superior frontal gyrus): 上前頭回，
            "ParaHippocampal_L": "左海馬傍回",  #  (Parahippocampal gyrus): 海馬傍回，
            "ParaHippocampal_R": "右海馬傍回",  #  (Parahippocampal gyrus): 海馬傍回，
            "Temporal_Inf_L": "左下側頭回",  #  (Inferior temporal gyrus): 下側頭回，
            "Temporal_Inf_R": "右下側頭回",  #  (Inferior temporal gyrus): 下側頭回，
            "Insula_L": "左島皮質",  # 島皮質，
            "Insula_R": "右島皮質",  # 島皮質，
            "Temporal_Mid_L": "左中側頭回",  # (Middle temporal gyrus): 中側頭回
            "Temporal_Mid_R": "右中側頭回",  # (Middle temporal gyrus): 中側頭回
        }

        def plot(df, y_label, atlas_name, model, model_name) -> None:
            df = df[:10].copy()
            # sns.set_context("paper", 1.6)
            sns.set_theme("paper", font="Noto Sans CJK JP", font_scale=1.6)  # type: ignore
            plt.style.use("ggplot")
            f, ax = plt.subplots(1, 1, dpi=150)
            if atlas_name == "aal":
                palette = aal_pallete(df[y_label], COLOR_DICT)
            else:
                palette = sns.color_palette("Set1")
            df[y_label] = df[y_label].replace(roi_jp_dict)
            ax = sns.barplot(y=y_label, x="percent", data=df, palette=palette, orient="h", ax=ax)
            """hatches = aal_hatches(df[y_label])
            for i, patch in enumerate(ax.patches): patch.set_hatch(hatches[i])"""
            ax.set_ylabel("ROI", fontsize=20)
            ax.set_xlabel("平均占有割合", fontsize=20)
            # ax.set_yticklabels([str(lab).replace("_", " ") for lab in df[y_label]])
            ax.set_title(f"{model_name}", fontsize=26)
            plt.savefig(f"./results/{model}/pereira2018_{model}_{atlas_name}_rois.png", bbox_inches="tight")
            plt.savefig(f"./results/{model}/pereira2018_{model}_{atlas_name}_rois.svg", bbox_inches="tight")

        plot(results_df_aal, "index", "aal", model, model_name)
        plot(results_df_gordon, "network", "gordon", model, model_name)


if __name__ == "__main__":
    main()
