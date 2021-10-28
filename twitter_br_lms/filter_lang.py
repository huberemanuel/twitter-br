import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import fasttext
import pandas as pd
from pkg_resources import resource_filename
from tqdm.auto import tqdm

import twitter_br_lms


class LanguageIdentification:
    def __init__(self, pretrained_lang_model: str):
        """
        Loads the fasttext model from the given path.

        pretrained_lang_model: str
            Path to the model.
        """
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Predict language with fasttext model.

        Parameters
        ----------
        text: str
            Input text to language identification prediction.

        Returns
        -------
        Tuple[List[str], List[float]]
            Tuple containing the list of predicted languages and the list of prediction scores.
        """
        predictions = self.model.predict(text, k=2)
        return predictions


def is_pt(text: str, model: LanguageIdentification) -> bool:
    """
    Check if a given text was written in pt-br

    Paramters
    ---------
    text: str
        Single text data.
    model: LanguageIdentification
        Language identification model from fasttext.FastTex.

    Returns
    -------
    bool
        Whether the given text was written in pt-br or not.
    """
    text = text.replace("\n", " ").lower()
    lang, scores = model.predict_lang(text)
    if lang[0] != "__label__pt" and scores[0] > 0.9:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter messages on pt-br")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the input data that will be filtered",
        default=resource_filename(twitter_br_lms.__name__, "data/raw"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output dir where filtered CSVs will be stored",
        default=resource_filename(twitter_br_lms.__name__, "data/interim"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path to the default cache dir.",
        default="~/.cache/twitter-br",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of fasttext model to be found at `cache_dir`",
        default="lid.176.bin",
    )
    args = parser.parse_args()

    model_path = Path(args.cache_dir).joinpath(args.model_name)
    model = LanguageIdentification(str(model_path))
    base_path = Path(args.data_path)
    output_path = Path(args.output_path)
    if base_path.is_file():
        input_files = [base_path]
    else:
        input_files = list(base_path.glob("**/*.csv"))

    for input_file in tqdm(input_files, desc="Filtering portuguese tweets"):
        df = pd.read_csv(input_file, header=0, names=["text"])
        before_len = len(df)
        df = df[df["text"].apply(lambda x: is_pt(x, model))]
        after_len = len(df)
        file_basename = input_file.stem
        dir_name = input_file.parent.stem
        logging.info(
            "Removed {} tweets, a {:.2f} reduction for dataset {}".format(
                before_len - after_len,
                (1 - after_len / before_len),
                Path(dir_name).joinpath(file_basename),
            )
        )
        output_dir = output_path.joinpath(dir_name)
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir.joinpath("{}.csv".format(file_basename)), index=False, header=0)
