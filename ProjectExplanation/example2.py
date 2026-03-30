"""
EXAMPLE 2: Causal Masking — Why the Decoder Can't "Cheat"

The Transformer decoder generates tokens one by one.
But during TRAINING, we feed the whole target sequence at once for efficiency
(teacher forcing). This creates a problem: if the model at position 3 can
see "what comes at position 4", it's cheating — it would never learn to
actually predict the future.

The solution is the CAUSAL MASK (also called the "look-ahead mask").

This example shows:
  1. What a causal mask looks like
  2. How it is used inside attention
  3. What happens WITHOUT a mask (the model cheats)
  4. What happens WITH the mask (correct behaviour)

Run with: python example2.py
"""

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Visualise the causal mask
# ─────────────────────────────────────────────────────────────────────────────
# Taken directly from SMTModelForCausalLM._generate_causal_mask()
def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Returns a bool tensor of shape (seq_len, seq_len).
    True at position (i, j) means: "token i is NOT allowed to attend to token j".
    This is True only when j > i (future tokens).
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1
    )

SEQ_LEN = 5  # e.g. <bos> clefG2 <t> clefF4 <b>

mask = generate_causal_mask(SEQ_LEN)
print("=" * 60)
print("Causal Mask (True = BLOCKED from attending):")
print("Rows = query positions, Columns = key positions")
print()
print("     ", "  ".join(f"t{j}" for j in range(SEQ_LEN)))
for i in range(SEQ_LEN):
    row = "  ".join("block" if mask[i, j] else "  ok " for j in range(SEQ_LEN))
    print(f" t{i}: {row}")
print()
print("Token t2 can see: t0, t1, t2  — but NOT t3 or t4.")
print("Token t0 can see: only t0 itself.")

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Scaled dot-product attention, manually
# ─────────────────────────────────────────────────────────────────────────────
# The core attention formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
#
# Let's see how the mask sets future attention weights to zero.

D = 4  # tiny embedding dimension for illustration
torch.manual_seed(42)

# Pretend we have 5 tokens, each with a 4-dim embedding
Q = torch.randn(SEQ_LEN, D)   # Queries from the decoder tokens
K = torch.randn(SEQ_LEN, D)   # Keys from the decoder tokens (self-attention)
V = torch.randn(SEQ_LEN, D)   # Values from the decoder tokens

scale = D ** -0.5
scores = (Q @ K.T) * scale    # Shape: (5, 5)

print("\n" + "=" * 60)
print("Raw attention scores (before mask, before softmax):")
print("Each row = one token's score for every other token")
print(torch.round(scores, decimals=2))

# ── WITHOUT MASK ──────────────────────────────────────────────────────────────
attn_no_mask = F.softmax(scores, dim=-1)
print("\nAttention weights WITHOUT mask:")
print("(Note: each row sums to 1.0 — token t2 attends to ALL tokens including future)")
print(torch.round(attn_no_mask, decimals=3))

# ── WITH CAUSAL MASK ──────────────────────────────────────────────────────────
# The mask sets future positions to -infinity BEFORE softmax.
# e^(-inf) = 0, so those positions get zero attention weight.
masked_scores = scores.masked_fill(mask, float('-inf'))

attn_with_mask = F.softmax(masked_scores, dim=-1)
print("\nAttention weights WITH causal mask:")
print("(Token t2 now only attends to t0, t1, t2 — future positions are 0)")
print(torch.round(attn_with_mask, decimals=3))

# ─────────────────────────────────────────────────────────────────────────────
# Part 3: The output — what the decoder actually "sees"
# ─────────────────────────────────────────────────────────────────────────────
output_no_mask   = attn_no_mask @ V
output_with_mask = attn_with_mask @ V

print("\n" + "=" * 60)
print("Decoder output at position t2:")
print(f"  WITHOUT mask: {output_no_mask[2].data}  ← polluted by future tokens t3, t4")
print(f"  WITH mask:    {output_with_mask[2].data} ← only sees past and present")
print()
print("These are different! Without the mask, the model at training time")
print("sees the future answers and never truly learns to predict them.")
print()
print("This is exactly what SMTModelForCausalLM._generate_causal_mask() does,")
print("and why it is passed to the MultiHeadAttention attn_mask argument.")
print()
print("During INFERENCE (predict()), the sequence grows one token at a time,")
print("so a causal mask isn't strictly needed — the future doesn't exist yet.")
print("But using it makes the training and inference behaviour consistent.")
