"""
This file is inspired from Nix Goldowsky-Dill's adaption of the tokenizer in https://github.com/juand-r/tiny_tokenizer.
"""

from collections.abc import Generator, Iterable
from pathlib import Path

from datasets import DatasetDict, IterableDatasetDict, load_dataset
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, Replace
from tokenizers.normalizers import Sequence as NormSequence
from tokenizers.pre_tokenizers import Digits, Punctuation, Sequence, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

OUT_DIR = Path("tokenizer")


def clean_dataset(dataset_name: str, column_name: str) -> Generator[str, None, None]:
    """
    Load and clean the dataset for tokenizer training. We assume that every dataset has some sort of
    train and validation split.
    """
    print(f"Loading and cleaning dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, trust_remote_code=False, streaming=True)
    trans = str.maketrans(
        {"\u201d": '"', "\u201c": '"', "\u2019": "'", "\u2018": "'", "\u2014": "-", "\u2026": "..."}
    )

    # Check if dataset has multiple splits or no split
    if isinstance(dataset, DatasetDict | IterableDatasetDict):
        for split_name, split_data in dataset.items():
            print(f"Processing {split_name} split...")
            for story in split_data:
                if isinstance(story, dict) and column_name in story:
                    yield (
                        story[column_name]
                        .translate(trans)
                        .encode("ascii", "ignore")
                        .decode("ascii")
                        .lower()
                    )
    else:
        print("Processing single split...")
        for story in dataset:
            if isinstance(story, dict) and column_name in story:
                yield (
                    story[column_name]
                    .translate(trans)
                    .encode("ascii", "ignore")
                    .decode("ascii")
                    .lower()
                )


def create_tokenizer(vocab_size: int) -> Tokenizer:
    """
    Create a tokenizer with integrated affix handling using Split pre-tokenizers.

    Args:
        vocab_size: The target vocabulary size for the tokenizer

    Returns:
        A configured Tokenizer object ready for training
    """
    print(f"Creating tokenizer with target vocabulary size: {vocab_size}")

    # Initialize WordPiece tokenizer
    tokenizer = Tokenizer(
        WordPiece(
            unk_token="[UNK]"  # type: ignore
        )
    )

    # Set normalizers (lowercase everything)
    tokenizer.normalizer = NormSequence(
        [
            Lowercase(),
            Replace("``", '"'),
            Replace("''", '"'),
        ]  # type: ignore
    )

    # Set up the pre-tokenizer sequence
    tokenizer.pre_tokenizer = Sequence(
        [Whitespace(), Punctuation(), Digits(individual_digits=True)]
    )  # type: ignore

    # Add post-processor for special tokens
    tokenizer.post_processor = TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 1)])  # type: ignore

    tokenizer.decoder = WordPieceDecoder(prefix="##")  # type: ignore

    return tokenizer


def train_tokenizer(data: Iterable[str], vocab_size: int) -> Tokenizer:
    """
    Train the tokenizer with the specified vocabulary size and cleaned data.

    Note: [UNK] token needs to be specified in two places:
    1. In special_tokens list - tells the trainer to include it in vocabulary
    2. Already specified in WordPiece constructor - tells the model which token to
    use for unknown words

    Args:
        data: List of cleaned text strings to train on
        vocab_size: The target vocabulary size

    Returns:
        Trained Tokenizer object
    """

    tokenizer = create_tokenizer(vocab_size)

    special_tokens = ["[UNK]", "[EOS]"]
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    tokenizer.train_from_iterator(data, trainer=trainer)
    print("Tokenizer training completed")

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, tokenizer_name: str) -> str:
    """
    Save tokenizer to file.

    Args:
        tokenizer: The tokenizer to save
        tokenizer_name: The filename for the tokenizer

    Returns:
        The full path where tokenizer was saved
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = f"{OUT_DIR}/{tokenizer_name}"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer_path


def prune_tokenizer(data: Iterable[str], tokenizer: Tokenizer) -> Tokenizer:
    """
    Prune tokenizer by removing unused tokens and reordering IDs sequentially.
    Note: [UNK] token is handled automatically by WordPiece constructor,
    so it's excluded from add_special_tokens() to avoid duplication.

    Args:
        tokenizer: Trained tokenizer object
        dataset_texts: List of text strings to check token usage against

    Returns:
        Pruned Tokenizer object with sequential IDs
    """
    original_vocab_size = len(tokenizer.get_vocab())
    print(f"Original vocabulary size: {original_vocab_size}")

    # Always keep special tokens
    special_tokens = get_special_token_ids(tokenizer)

    # Find used tokens in dataset
    used_token_ids = set()

    for text in tqdm(data, desc="Tokenizing dataset"):
        encoded = tokenizer.encode(text)
        used_token_ids.update(encoded.ids)

    # Keep both used and special tokens
    all_needed_tokens = used_token_ids | special_tokens

    print(f"Used tokens: {len(all_needed_tokens)}")
    print(f"Removing: {original_vocab_size - len(all_needed_tokens)} tokens")

    if len(all_needed_tokens) == original_vocab_size:
        print("No tokens to remove!")
        return tokenizer

    # Create new vocabulary with sequential IDs
    new_vocab = {}
    for new_id, old_id in enumerate(sorted(all_needed_tokens)):
        token_text = tokenizer.id_to_token(old_id)
        new_vocab[token_text] = new_id

    print(f"New vocabulary size: {len(new_vocab)}")

    # Create new tokenizer
    new_tokenizer: Tokenizer = Tokenizer(WordPiece(vocab=new_vocab, unk_token="[UNK]"))  # type: ignore

    # Add back all special tokens (except UNK which is handled by WordPiece constructor)
    special_tokens_to_add = []
    for token_id in special_tokens:
        token_text = tokenizer.id_to_token(token_id)
        if token_text != "[UNK]":
            special_tokens_to_add.append(AddedToken(token_text, special=True))

    if special_tokens_to_add:
        new_tokenizer.add_special_tokens(special_tokens_to_add)

    # Copy settings from original
    new_tokenizer.normalizer = tokenizer.normalizer  # type: ignore
    new_tokenizer.pre_tokenizer = tokenizer.pre_tokenizer  # type: ignore
    new_tokenizer.post_processor = tokenizer.post_processor  # type: ignore
    new_tokenizer.decoder = tokenizer.decoder  # type: ignore

    return new_tokenizer


def load_tokenizer(tokenizer_name: str) -> Tokenizer:
    """
    Load a tokenizer from file.

    Args:
        tokenizer_name: The filename of the tokenizer to load
    """
    return Tokenizer.from_file(f"{OUT_DIR}/{tokenizer_name}")


def get_special_token_ids(tokenizer: Tokenizer) -> set[int]:
    """Get IDs of all added special tokens automatically."""
    special_token_ids = set()

    # Get all added tokens and check for special tokens
    for added_token in tokenizer.get_added_tokens_decoder().values():
        if added_token.special:
            token_id = tokenizer.token_to_id(str(added_token))
            if token_id is not None:
                special_token_ids.add(token_id)

    # Always include UNK token ID
    unk_id = tokenizer.token_to_id("[UNK]")
    if unk_id is not None:
        special_token_ids.add(unk_id)

    return special_token_ids


def convert_to_hf_tokenizer(tokenizer: Tokenizer, model_max_length: int):
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[UNK]",  # Using UNK as pad since no dedicated PAD token
        model_max_length=model_max_length,
    )

    return hf_tokenizer


if __name__ == "__main__":
    vocab_size = 4096
    dataset_name = "SimpleStories/SimpleStories"
    column_name = "story"
    save_name = "simplestories-tokenizer.json"

    cleaned_data = clean_dataset(dataset_name, column_name)
    tokenizer = train_tokenizer(cleaned_data, vocab_size=vocab_size)

    # Create fresh iterator since generators can only be consumed once
    cleaned_data = clean_dataset(dataset_name, column_name)
    pruned_tokenizer = prune_tokenizer(cleaned_data, tokenizer)

    save_tokenizer(pruned_tokenizer, save_name)
