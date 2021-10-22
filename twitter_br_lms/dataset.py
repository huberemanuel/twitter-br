from typing import Dict, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LMDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        n_samples: Optional[int] = None,
        uncased: bool = True,
        debugging: bool = False,
    ):
        """
        Initialize LMDataset, loading all samples into the RAM.

        Parameters
        ---------
        base_path: str
            Path to the actual CSV file or to the base folder of the project,
            in ths case is expected to find a `processed` dir insise base_path.
        tokenizer: Optional[AutoTokenizer]
            Tokenizer used on sample retrievals.
        n_samples: Optional[int]
            If passed, only retrieves n_samples samples from the dataset.
        uncased: bool
            Wheter lowercase samples or not.
        debugging: bool
            Wheter load only 100 data samples for debugging purposes.
        """
        self.tokenizer = tokenizer
        df = pd.read_csv(file_path, header=0, names=["text"], nrows=100 if debugging else None)

        if uncased:
            df["text"] = df["text"].str.lower()

        if n_samples:
            df = df.sample(n_samples)

        self.samples = df["text"].to_list()

    def __len__(self):
        return len(self.samples)

    def _tokenize(
        self,
        text: str,
        padding: Union[str, bool] = False,
        max_seq_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a single `text`

        Parameters
        ----------
        text: str
            A single sample of text.
        padding: Union[str, bool]
            Padding type of the folling list ["longest", "max_length", "do_not_pad", False].
        max_seq_length: Optional[int]
            Trims the number of tokens with given parameter.

        Returns
        -------
        Dict[str, torch.Tensor]
            input_ids and mask.
        """
        return self.tokenizer(
            text,
            padding=padding,
            truncation=True,
            max_length=max_seq_length or self.tokenizer.model_max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

    def __getitem__(
        self,
        i,
        padding: Optional[Union[str, bool]] = False,
        max_seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Retrives a single preprocessed and tokenized item.

        Parameters
        ----------
        padding: Union[str, bool]
            Padding type of the folling list ["longest", "max_length", "do_not_pad", False].
        max_seq_length: Optional[int]
            Trims the number of tokens with given parameter.

        Returns
        -------
        torch.Tensor
            Tensor containing all token ids.
        """
        if self.tokenizer:
            return self._tokenize(self.samples[i], padding, max_seq_length)["input_ids"].squeeze()
        return self.samples[i]
