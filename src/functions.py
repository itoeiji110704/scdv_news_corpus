from typing import Dict

import numpy as np
import pandas as pd
from sklearn import manifold, model_selection
from xgboost import XGBClassifier

from src.utils import logger, timeit


@timeit
def to_tsne_df(in_vec: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Calculate t-SNE on document vector and save as a data frame.

    Args:
        in_vec (np.ndarray): document vector.
        targets (np.ndarray): target (category) list.

    Returns:
        pd.DataFrame: t-SNE vector data frame.
    """
    tsne_vec = manifold.TSNE(n_components=2).fit_transform(in_vec)
    df_tsne = pd.DataFrame(
        {
            "x": tsne_vec[:, 0],
            "y": tsne_vec[:, 1],
            "category": targets,
        }
    )
    return df_tsne


@timeit
def compare_xgboost_accuracy(
    targets: np.ndarray, dic_vecs: Dict[str, pd.DataFrame], cv_trial_num: int = 8
) -> pd.DataFrame:
    """Get data frame of compare model classification accuracy using each document vectors.

    Args:
        targets (np.ndarray): target (category) list.
        dic_vecs (Dict[str, pd.DataFrame]): document vectors.
        cv_trial_num (int, optional): the number of CV. Defaults to 8.

    Returns:
        pd.DataFrame: a data frame of result.
    """
    model = XGBClassifier()

    df_compare = pd.DataFrame(
        columns=["method", "train_accuracy", "test_accuracy", "fit_time"]
    )
    scoring = ["accuracy"]

    for method, vecs in dic_vecs.items():

        logger.info(f"Train model using {method}")
        cv_rlts = model_selection.cross_validate(
            model,
            vecs,
            targets,
            scoring=scoring,
            cv=cv_trial_num,
            return_train_score=True,
        )
        for i in range(cv_trial_num):
            s = pd.Series(
                [
                    method,
                    cv_rlts["train_accuracy"][i],
                    cv_rlts["test_accuracy"][i],
                    cv_rlts["fit_time"][i],
                ],
                index=df_compare.columns,
                name=method + str(i),
            )
            df_compare = df_compare.append(s)

    return df_compare
