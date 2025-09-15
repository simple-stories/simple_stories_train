"""Simple test for tokenizer pruning functionality."""

from simple_stories_train.tokenizer import convert_to_hf_tokenizer, prune_tokenizer, train_tokenizer


def create_test_tokenizer():
    """Create a fresh tokenizer for testing."""
    train_data = ["hello world", "hello there", "world peace", "simple stories"]
    return train_tokenizer(iter(train_data), vocab_size=200)


def test_special_tokens_preserved():
    """Verify special tokens exist and are unique in both original and pruned tokenizers."""
    tokenizer = create_test_tokenizer()

    vocab_orig = tokenizer.get_vocab()
    assert "[UNK]" in vocab_orig and "[EOS]" in vocab_orig
    assert vocab_orig["[UNK]"] in [0, 1] and vocab_orig["[EOS]"] in [0, 1]
    assert vocab_orig["[UNK]"] != vocab_orig["[EOS]"]

    test_data = ["hello world"]
    pruned = prune_tokenizer(iter(test_data), tokenizer)
    vocab_pruned = pruned.get_vocab()
    assert "[UNK]" in vocab_pruned and "[EOS]" in vocab_pruned
    assert vocab_pruned["[UNK]"] in [0, 1] and vocab_pruned["[EOS]"] in [0, 1]
    assert vocab_pruned["[UNK]"] != vocab_pruned["[EOS]"]


def test_unused_tokens_removed():
    """Verify pruning removes unused tokens from vocabulary."""
    tokenizer = create_test_tokenizer()

    original_size = len(tokenizer.get_vocab())
    test_data = ["hello"]

    pruned = prune_tokenizer(iter(test_data), tokenizer)

    assert len(pruned.get_vocab()) < original_size


def test_functionality_preserved():
    """Verify encode/decode functionality works before and after pruning."""
    tokenizer = create_test_tokenizer()

    encoded_orig = tokenizer.encode("hello world")
    decoded_orig = tokenizer.decode(encoded_orig.ids)
    assert decoded_orig == "hello world"

    test_data = ["hello world"]
    pruned = prune_tokenizer(iter(test_data), tokenizer)
    encoded_pruned = pruned.encode("hello world")
    decoded_pruned = pruned.decode(encoded_pruned.ids)
    assert decoded_pruned == "hello world"


def test_sequential_ids():
    """Verify pruned tokenizer has sequential token IDs starting from 0."""
    tokenizer = create_test_tokenizer()

    token_ids_orig = sorted(tokenizer.get_vocab().values())
    assert token_ids_orig[0] >= 0
    assert len(token_ids_orig) == len(set(token_ids_orig))

    test_data = ["hello"]
    pruned = prune_tokenizer(iter(test_data), tokenizer)
    token_ids_pruned = sorted(pruned.get_vocab().values())
    assert token_ids_pruned == list(range(len(token_ids_pruned)))


def test_eos_appended():
    """Verify EOS token is appended as last token before and after pruning."""
    tokenizer = create_test_tokenizer()

    eos_id_orig = tokenizer.token_to_id("[EOS]")
    encoded_orig = tokenizer.encode("hello world")
    assert encoded_orig.ids[-1] == eos_id_orig

    test_data = ["hello world"]
    pruned = prune_tokenizer(iter(test_data), tokenizer)
    eos_id_pruned = pruned.token_to_id("[EOS]")
    encoded_pruned = pruned.encode("hello world")
    assert encoded_pruned.ids[-1] == eos_id_pruned


def test_unk_for_unknown_words():
    """Verify UNK token is used for unknown words before and after pruning."""
    tokenizer = create_test_tokenizer()
    unk_id_orig = tokenizer.token_to_id("[UNK]")
    encoded_orig = tokenizer.encode("antidisestablishmentarianism")
    assert unk_id_orig in encoded_orig.ids

    test_data = ["hello world"]
    pruned = prune_tokenizer(iter(test_data), tokenizer)
    unk_id_pruned = pruned.token_to_id("[UNK]")
    encoded_pruned = pruned.encode("antidisestablishmentarianism")
    assert unk_id_pruned in encoded_pruned.ids


def test_convert_to_hf_tokenize():
    """Verify conversion to HF tokenizer produces identical token IDs."""
    original_tokenizer = create_test_tokenizer()
    hf_tokenizer = convert_to_hf_tokenizer(original_tokenizer, model_max_length=512)

    test_strings = ["hello world", "hello there", "antidisestablishmentarianism"]

    for test_str in test_strings:
        orig_ids = original_tokenizer.encode(test_str).ids
        hf_ids = hf_tokenizer.encode(test_str)

        assert orig_ids == hf_ids, f"Token IDs differ for '{test_str}': {orig_ids} vs {hf_ids}"
