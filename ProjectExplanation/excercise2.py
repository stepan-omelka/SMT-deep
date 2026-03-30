"""
EXERCISE 2: Build the Vocabulary from a Set of Transcriptions

★ GOAL: Understand how the SMT vocabulary (w2i / i2w) is built.
        This vocabulary maps musical token strings ("4c", "clefG2", ...)
        to unique integer indices — because the model only works with numbers.

        This is EXACTLY what utils.make_vocabulary() does, which is called
        every time you start a new training run.

★ YOUR TASK: Implement the three functions marked with TODO.
             You may NOT call check_and_retrieveVocabulary or make_vocabulary
             from utils.py — replicate the logic yourself.

★ EXPECTED RESULT when correct:
        Vocabulary size: 15  (or similar, depending on the sample data)
        <pad> is always index 0
        Round-trip check passed: every token survives w2i → i2w

★ BONUS: After completing, answer the questions at the bottom.

Run with: python excercise2.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Sample data — 3 short bekern transcriptions (simplified for the exercise)
# In real training, there are thousands of these from the HuggingFace dataset.
# ─────────────────────────────────────────────────────────────────────────────
TRANSCRIPTIONS = [
    # transcription 1
    ["<bos>", "clefG2", "<t>", "clefF4", "<b>", "*k[]", "<t>", "*k[]", "<b>",
     "4c", "<t>", "4E", "<b>", "4d", "<t>", "4F", "<b>", "<eos>"],
    # transcription 2
    ["<bos>", "clefG2", "<t>", "clefF4", "<b>", "*k[f#]", "<t>", "*k[f#]", "<b>",
     "4g", "<t>", "4G", "<b>", "<eos>"],
    # transcription 3 — introduces new tokens
    ["<bos>", "clefG2", "<t>", "clefF4", "<b>",
     "8c", "<t>", "8E", "<b>", "8d", "<t>", "8F", "<b>",
     "4r", "<t>", "4r", "<b>", "<eos>"],
]


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: Collect all unique tokens from all transcriptions
# ─────────────────────────────────────────────────────────────────────────────
def collect_unique_tokens(transcriptions: list[list[str]]) -> set[str]:
    """
    Given a list of transcriptions (each is a list of string tokens),
    return a Python SET of all unique tokens across all transcriptions.

    HINT: Use a set. Iterate through all transcriptions and all tokens.
          Do NOT include '<pad>' here — it will be added separately.

    Example:
        collect_unique_tokens([["<bos>", "4c"], ["<bos>", "4d"]])
        → {"<bos>", "4c", "4d"}
    """
    # TODO: implement this
    raise NotImplementedError("Implement collect_unique_tokens()")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: Build the w2i and i2w dictionaries
# ─────────────────────────────────────────────────────────────────────────────
def build_vocab(unique_tokens: set[str]) -> tuple[dict, dict]:
    """
    Given a set of unique tokens, return two dictionaries:
        w2i: token string → integer index (word to index)
        i2w: integer index → token string (index to word)

    Rules (same as utils.make_vocabulary):
        - '<pad>' must ALWAYS be at index 0.
        - All other tokens get indices starting from 1.
        - The order of the other tokens does not matter (they come from a set).

    HINT: Enumerate over unique_tokens. Use idx+1 as the index so idx 0 is free.
          Then manually add '<pad>' → 0 and 0 → '<pad>'.

    Example:
        build_vocab({"<bos>", "4c"})
        → w2i = {"<pad>": 0, "<bos>": 1, "4c": 2}   (or reversed for bos/4c)
           i2w = {0: "<pad>", 1: "<bos>", 2: "4c"}
    """
    # TODO: implement this
    raise NotImplementedError("Implement build_vocab()")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: Tokenize a new transcription using the vocabulary
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(transcription: list[str], w2i: dict) -> list[int]:
    """
    Convert a list of token strings into a list of integer indices using w2i.

    If a token is NOT in w2i (out-of-vocabulary), skip it
    (in the real SMT, this shouldn't happen because the vocab is
    built over all splits, but it's good practice to handle it).

    Example:
        tokenize(["<bos>", "4c", "<eos>"], {"<pad>":0, "<bos>":1, "4c":2, "<eos>":3})
        → [1, 2, 3]
    """
    # TODO: implement this
    raise NotImplementedError("Implement tokenize()")


# ─────────────────────────────────────────────────────────────────────────────
# Main: run your implementations and verify them
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1
    unique_tokens = collect_unique_tokens(TRANSCRIPTIONS)
    print(f"Unique tokens found: {len(unique_tokens)}")

    # Step 2
    w2i, i2w = build_vocab(unique_tokens)
    print(f"Vocabulary size (including <pad>): {len(w2i)}")
    print(f"Index of '<pad>':   {w2i['<pad>']}  (must be 0)")
    print(f"Token at index 0:   '{i2w[0]}'      (must be '<pad>')")

    # Step 3: tokenize a sample
    sample = TRANSCRIPTIONS[0]
    indices = tokenize(sample, w2i)
    print(f"\nFirst transcription tokenized:")
    print(f"  Tokens:  {sample[:5]} ...")
    print(f"  Indices: {indices[:5]} ...")

    # Round-trip check: decode back and compare
    decoded = [i2w[idx] for idx in indices]
    assert decoded == sample, f"Round-trip FAILED!\n  Original: {sample}\n  Decoded:  {decoded}"
    print("\n✓ Round-trip check passed: tokenize → detokenize gives the original sequence.")

    # Extra: show a portion of the vocabulary
    print("\nFirst 8 entries of the vocabulary:")
    for idx in range(min(8, len(i2w))):
        print(f"  {idx:>3} → '{i2w[idx]}'")

# ─────────────────────────────────────────────────────────────────────────────
# BONUS QUESTIONS (think about these — no code required):
#
# Q1: Why is <pad> always index 0?
#     HINT: Look at nn.CrossEntropyLoss(ignore_index=0) in smt_trainer.py
#           and nn.Embedding(vocab_size, d_model, padding_idx=0) in the models.
#           What would break if <pad> were index 5 instead?
#
# Q2: The vocabulary is built from the UNION of train + val + test tokens.
#     Why not build it only from the training set?
#     What would happen at evaluation time if val/test had unseen tokens?
#
# Q3: In utils.make_vocabulary(), the vocabulary order depends on Python's set,
#     which is NOT deterministic across runs. The code saves w2i and i2w to
#     .npy files so the same vocab is reused. Why is this critical?
#     HINT: think about what happens to stored model weights if indices change.
# ─────────────────────────────────────────────────────────────────────────────
