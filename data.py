from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from torch.utils.data import Dataset, DataLoader
import torch

from pathlib import Path


def sentence_generator(dataset, language: str):
    for data in dataset:
        yield data["translation"][language]

def create_or_load_tokenizer(dataset, language: str, file_path:str=None):
    if file_path is not None:
        if Path(file_path).exists():
            tokenizer = Tokenizer.from_file(file_path)
            return tokenizer
    
    trainer = WordLevelTrainer(
        min_frequency=2,
        show_progress=True,
        special_tokens=["<SOS>", "<EOS>", "<PAD>", "<unk>"],
    )
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(sentence_generator(dataset, language), trainer=trainer)
    tokenizer.save(file_path)
    return tokenizer


def is_within_length_limit(
    example,
    source_tokenizer,
    target_tokenizer,
    source_language,
    target_language,
    max_source_content_tokens,
    max_target_content_tokens
):
    source_len = len(source_tokenizer.encode(example["translation"][source_language]).ids)
    target_len = len(target_tokenizer.encode(example["translation"][target_language]).ids)
    return source_len <= max_source_content_tokens and target_len <= max_target_content_tokens

    
def train_tokenizers(
    dataset_path:str,
    dataset_name:str,
    source_language: str,
    target_language: str,
    source_file_path:str,
    target_file_path:str,
    max_seq_len: int,
    hf_token: str = None,
):
    dataset = load_dataset(
        path=dataset_path,
        name=dataset_name,
        split="train",
        token=hf_token,
    )
    source_tokenizer = create_or_load_tokenizer(dataset, source_language, source_file_path)
    target_tokenizer = create_or_load_tokenizer(dataset, target_language, target_file_path)

    original_size = len(dataset)
    filtered_dataset = dataset.filter(
        is_within_length_limit,
        fn_kwargs={
            "source_tokenizer": source_tokenizer,
            "target_tokenizer": target_tokenizer,
            "source_language": source_language,
            "target_language": target_language,
            "max_source_content_tokens": max_seq_len - 2, # leaving room for SOS and EOS
            "max_target_content_tokens": max_seq_len - 1, # leaving room for EOS
        }
    )
    filtered_size = len(filtered_dataset)
    
    print(f"Filtered dataset: {filtered_size:,} / {original_size:,} pairs kept "
          f"({100 * filtered_size / original_size:.1f}%)")

    max_source_seq_len, max_target_seq_len = 0, 0
    for data in filtered_dataset:
        source_token_ids = source_tokenizer.encode(data["translation"][source_language]).ids
        target_token_ids = target_tokenizer.encode(data["translation"][target_language]).ids
        
        max_source_seq_len = max(max_source_seq_len, len(source_token_ids))
        max_target_seq_len = max(max_target_seq_len, len(target_token_ids))

    data = {
        "source_tokenizer": source_tokenizer,
        "target_tokenizer": target_tokenizer,
        "source_seq_len": max_source_seq_len + 2,
        "target_seq_len": max_target_seq_len + 1,
        "source_vocab_size": source_tokenizer.get_vocab_size(),
        "target_vocab_size": target_tokenizer.get_vocab_size(),
    }

    return data


def create_data_loader(
    dataset_path:str,
    dataset_name:str,
    split:str,
    trained_tokens_dict: dict,
    source_language: str,
    target_language: str,
    batch_size: int,
    shuffle: bool,
    hf_token: str = None,
):
    dataset = load_dataset(
        path=dataset_path,
        name=dataset_name,
        split=split,
        token=hf_token,
    )

    filtered_dataset = dataset.filter(
        is_within_length_limit,
        fn_kwargs={
            "source_tokenizer": trained_tokens_dict["source_tokenizer"],
            "target_tokenizer": trained_tokens_dict["target_tokenizer"],
            "source_language": source_language,
            "target_language": target_language,
            "max_source_content_tokens": trained_tokens_dict["source_seq_len"] - 2, # leaving room for SOS and EOS
            "max_target_content_tokens": trained_tokens_dict["target_seq_len"] - 1, # leaving room for EOS
        }
    )

    translation_dataset = TranslationDataset(
        data=filtered_dataset,
        source_seq_len=trained_tokens_dict["source_seq_len"],
        target_seq_len=trained_tokens_dict["target_seq_len"],
        source_language=source_language,
        target_language=target_language,
        source_tokenizer=trained_tokens_dict["source_tokenizer"],
        target_tokenizer=trained_tokens_dict["target_tokenizer"],
    )
    
    data_loader = DataLoader(
        dataset=translation_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return data_loader


class TranslationDataset(Dataset):
    def __init__(
        self,
        data,
        source_seq_len,
        target_seq_len,
        source_language: str,
        target_language: str,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer
    ):
        super().__init__()
        self.data = data
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.source_language = source_language
        self.target_language = target_language
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_vocab_size = source_tokenizer.get_vocab_size()
        self.target_vocab_size = target_tokenizer.get_vocab_size()

    def __len__(self):
        return len(self.data)

    @staticmethod
    def create_token_and_mask(
        sentence: str,
        tokenizer: Tokenizer,
        seq_len: int,
        token_type: str
    ):
        mask = None
        token_ids = tokenizer.encode(sentence).ids

        sos_token_id_ls, eos_token_id_ls = [], []
        if token_type != "label":
            sos_token_id_ls = [tokenizer.token_to_id("<SOS>")]
        if token_type != "decoder":
            eos_token_id_ls = [tokenizer.token_to_id("<EOS>")]
        
        token_ids = sos_token_id_ls + token_ids + eos_token_id_ls
        if len(token_ids) > seq_len:
            raise ValueError(f"seq_len: {seq_len}, token_ids length: {len(token_ids)}")

        pad_token_id = tokenizer.token_to_id("<PAD>")
        token_ids = torch.tensor(token_ids + [pad_token_id] * (seq_len - len(token_ids)), dtype=torch.int64)
        
        padding_mask = torch.ones(len(token_ids))
        padding_mask[token_ids == pad_token_id] = 0

        causal_mask = torch.tril(torch.ones(seq_len, seq_len))

        if token_type == "encoder":
            mask = padding_mask # 
        elif token_type == "decoder":
            # Pytorch broadcasts the padding mask across all rows of the causal_mask. Each row of causal_mask is multiplied by padding_mask
            mask = padding_mask.unsqueeze(0) * causal_mask # (1, seq_len) * (seq_len, seq_len)
        
        return token_ids, mask

    def __getitem__(self, idx):
        pair = self.data[idx]
        source_sentence = pair["translation"][self.source_language]
        source_token_ids, padding_mask = TranslationDataset.create_token_and_mask(
            sentence=source_sentence,
            tokenizer=self.source_tokenizer,
            seq_len=self.source_seq_len,
            token_type="encoder",
        )
        target_sentence = pair["translation"][self.target_language]
        target_token_ids, padding_causal_mask = TranslationDataset.create_token_and_mask(
            sentence=target_sentence,
            tokenizer=self.target_tokenizer,
            seq_len=self.target_seq_len,
            token_type="decoder",
        )
        label_token_ids, _ = TranslationDataset.create_token_and_mask(
            sentence=target_sentence,
            tokenizer=self.target_tokenizer,
            seq_len=self.target_seq_len,
            token_type="label",
        )
        
        data = {
            "source_sentence": source_sentence,
            "target_sentence": target_sentence,
            "encoder_input": source_token_ids,
            "decoder_input": target_token_ids,
            "label": label_token_ids,
            "encoder_mask": padding_mask.unsqueeze(0).unsqueeze(0),
            "decoder_mask": padding_causal_mask.unsqueeze(0),
        }

        return data