import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from twitter_br_lms.args import SmartFormatter


MAX_TWEETS_DATASET = 30_000_000  # Max tweets to get from a single file.


def main():
    parser = argparse.ArgumentParser(
        "Split interim datasets into train and validation sets", formatter_class=SmartFormatter
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="""R|Path to the input data. The directory should have the following structure:
        data_path/
            dataset1/
                train.csv
            dataset2/
                file.csv
            datasetn/
                random_name.csv""",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path that processed CSVs are going to be stored.",
        default=".",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        help="Fractino of the dataset to be set as the training set. The (1 `train_frac`)"
        " will be used as the test size.",
        default=0.9,
    )
    parser.add_argument(
        "--drop_duplicates",
        action="store_true",
        default=False,
        help="If set the pandas.drop_duplicates will be executed, this may take a while to finish",
    )
    parser.add_argument(
        "--seed", type=int, help="Default seed used in pandas random state", default=42
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    if not data_path.exists():
        raise ValueError("data_path {} does not exists".format(args.data_path))

    if not output_path.exists():
        raise ValueError("output_path {} does not exists".format(args.output_path))

    input_files = list(data_path.glob("**/*.csv"))

    samples = []

    for input_file in tqdm(input_files, desc="Splitting interim data into train and val sets"):
        df = pd.read_csv(input_file, header=0, names=["text"])

        if len(df) > MAX_TWEETS_DATASET:
            df = df.sample(MAX_TWEETS_DATASET)

        samples += df["text"].to_list()

    df = pd.DataFrame(samples, columns=["text"])

    if args.drop_duplicates:
        print("Dropping duplicates... go grab a ☕")
        df = df.drop_duplicates(subset=["text"])

    train_df = df.sample(frac=args.train_frac, random_state=args.seed)
    val_df = df.drop(train_df.index)

    train_df.to_csv(output_path.joinpath("train.csv"), index=None, header=0)

    val_df.to_csv(output_path.joinpath("val.csv"), index=None, header=0)


if __name__ == "__main__":
    main()
