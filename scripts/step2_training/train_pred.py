import time
from typing import Tuple, Dict

import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, make_pipeline


def predict_latent(
    inputs: Dict[str, np.ndarray],
    reshape: bool = False,
    feature_selection: str = "anova",
    regressor: str = "ridge",
) -> Tuple[np.ndarray, Pipeline]:
    pipeline_step = []

    if reshape == True:
        inputs["X_train"] = inputs["X_train"].reshape(len(inputs["X_train"]), -1)
        inputs["X_test"] = inputs["X_test"].reshape(len(inputs["X_test"]), -1)
    elif reshape == False:
        pass
    else:
        raise ValueError("reshape should be True or Flase")

    if feature_selection == "none":
        pass
    else:
        raise ValueError("feature selection option is wrong.")

    if regressor == "ridge":
        pipeline_step.append(RidgeCV(alphas=np.logspace(-5, 5, 20)))
    else:
        raise ValueError("regressor option is wrong.")

    pipeline = make_pipeline(*pipeline_step)
    print("pipeline:", pipeline)
    pipeline.fit(inputs["X_train"], inputs["Y_train"])
    Y_pred = pipeline.predict(inputs["X_test"])
    print("R2 score:", r2_score(inputs["Y_test"], Y_pred))
    assert isinstance(Y_pred, np.ndarray)

    # with open(f'./pipeline_{feature_selection}_{regressor}.pickle', mode='wb') as f: pickle.dump(pipeline, f)

    return Y_pred, pipeline


def run_test_case():
    X, Y = make_regression(n_samples=100, n_features=1000, n_targets=768, random_state=0)  # type: ignore
    X = X.reshape(-1, 10, 10, 10)
    print("example dataset:", X.shape, Y.shape)

    feature_selection = "none"
    regressor = "ridge"
    reshape = True

    start = time.time()
    Y_pred, pipeline = predict_latent(
        {
            "X_train": X[:-10],
            "Y_train": Y[:-10],
            "X_test": X[-10:],
            "Y_test": Y[-10:],
        },
        reshape=reshape,
        feature_selection=feature_selection,
        regressor=regressor,
    )
    print(pipeline["ridgecv"].coef_.shape)  # type: ignore
    print(f"time: {time.time() - start:.2f} s")


if __name__ == "__main__":
    run_test_case()
