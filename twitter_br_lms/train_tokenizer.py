import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

from twitter_br_lms.dataset import LMDataset


def train_roberta(train_data: DataLoader, output_path: str, vocab_size: int):

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train_from_iterator(
        train_data,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )

    # Save files to disk
    Path(output_path).mkdir(exist_ok=True)
    tokenizer.save_model(output_path, "roberta")


def main():
    parser = argparse.ArgumentParser("Train a tokenizer for further mlm training.")
    parser.add_argument(
        "--train_file",
        type=str,
        help="Path to the training CSV file.",
    )
    parser.add_argument("--output_path", type=str, help="Output path the tokenizer will be saved.", default=".")
    parser.add_argument(
        "--tokenizer", type=str, help='Type of the tokenizer from the following list ["roberta"]', default="roberta"
    )
    parser.add_argument("--vocab_size", type=int, help="Size of the vocabulary", default=52_000)
    parser.add_argument(
        "--sanity_debug", action="store_true", help="Trains a small tokenizer with 100 data samples", default=None
    )
    parser.add_argument("--uncased", action="store_true", help="Whether consider word case or not.", default=False)
    args = parser.parse_args()

    tokenizer_types = ["roberta"]

    if args.tokenizer not in tokenizer_types:
        raise ValueError("Tokenizer type {} not supported".format(args.tokenizer))

    # Load train data without a tokenizer
    train_data = LMDataset(args.train_file, tokenizer=None, uncased=args.uncased, debugging=args.sanity_debug)

    if args.tokenizer == "roberta":
        train_roberta(
            train_data,
            args.output_path,
            args.vocab_size,
        )


if __name__ == "__main__":
    main()
