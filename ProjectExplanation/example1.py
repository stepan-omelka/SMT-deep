"""
EXAMPLE 1: The Heart of Machine Learning — Loss, Backpropagation, and Gradient Descent

This example reproduces the EXACT learning mechanism used in SMT training.
Run it with: python example1.py

No GPUs needed. No datasets. Just the math.

What you will see:
  - A tiny neural network "learning" to classify tokens
  - The loss going DOWN each iteration (the model is getting better)
  - What gradients look like before and after an optimizer step
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
# SETUP: A minimal problem
# ─────────────────────────────────────────────────────────────────────────────
#
# Imagine we have a tiny vocabulary of 5 music symbols:
VOCAB = {
    0: "<pad>",
    1: "<bos>",
    2: "clefG2",   # treble clef
    3: "4c",       # quarter note C
    4: "<eos>",
}
VOCAB_SIZE = len(VOCAB)

# We have ONE training example:
#   Input token index: 1 (<bos>)
#   Correct next token: 2 (clefG2)
#   (After a <bos>, the score typically starts with a clef)
input_token_idx  = torch.tensor([1])    # <bos>
correct_next_idx = torch.tensor([2])    # clefG2  ← ground truth

# ─────────────────────────────────────────────────────────────────────────────
# MODEL: The simplest possible "next token predictor"
# ─────────────────────────────────────────────────────────────────────────────
# Embedding: token index → dense vector (like a lookup table)
# Linear:    dense vector → score for each vocab entry
class TinyPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj  = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_idx):
        x = self.embed(token_idx)   # shape: (1, embed_dim)
        return self.proj(x)         # shape: (1, vocab_size) — raw scores (logits)

model     = TinyPredictor(VOCAB_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# CrossEntropyLoss = Softmax + NegativeLogLikelihood in one step.
# It compares the model's probability distribution over vocabulary
# against the single correct token index.
loss_fn = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP — 30 iterations on this one example
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Training a model to predict  <bos> → '{VOCAB[2]}'")
print("=" * 60)
print(f"{'Iter':>5}  {'Loss':>10}  {'Model Prediction':>18}  {'Correct?':>8}")
print("-" * 60)

for i in range(30):
    # ── STEP 1: FORWARD PASS ──────────────────────────────────────────────────
    # Run the model. Get raw scores (logits) for each possible next token.
    logits = model(input_token_idx)   # shape: (1, 5)

    # ── STEP 2: COMPUTE LOSS ─────────────────────────────────────────────────
    # Cross-entropy measures how "surprised" the model is by the correct answer.
    # If the model assigns probability 0.99 to 'clefG2' → loss ≈ 0 (good)
    # If the model assigns probability 0.01 to 'clefG2' → loss ≈ 4.6 (very bad)
    loss = loss_fn(logits, correct_next_idx)

    # ── STEP 3: BACKWARD PASS (Backpropagation) ───────────────────────────────
    # PyTorch computes d(loss)/d(weight) for EVERY weight in the model.
    # This is "how much does this weight contribute to the mistake?"
    optimizer.zero_grad()   # Clear previous gradients (they accumulate otherwise!)
    loss.backward()         # Compute all gradients

    # ── STEP 4: OPTIMIZER STEP ────────────────────────────────────────────────
    # Adam uses the gradients to nudge every weight in the direction that
    # reduces the loss. LR=0.1 controls how big each nudge is.
    optimizer.step()

    # ── LOGGING ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        predicted_idx = logits.argmax(dim=-1).item()
        predicted_word = VOCAB[predicted_idx]
        correct = "✓" if predicted_idx == correct_next_idx.item() else "✗"

    print(f"{i+1:>5}  {loss.item():>10.4f}  {predicted_word:>18}  {correct:>8}")

# ─────────────────────────────────────────────────────────────────────────────
# SHOW WHAT GRADIENTS LOOK LIKE (on the last backward pass)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("What does a gradient look like? (embed layer, first row):")
print("  This is the direction we're nudging the <bos> embedding.")
print(f"  gradient = {model.embed.weight.grad[1].data}")
print()
print("After training, the model's top predictions for <bos>:")
model.eval()
with torch.no_grad():
    logits = model(input_token_idx)
    probs = torch.softmax(logits, dim=-1)
    for idx, prob in sorted(enumerate(probs[0]), key=lambda x: -x[1]):
        print(f"  {VOCAB[idx]:>10}  →  {prob.item():.4f}")

print()
print("Notice 'clefG2' should have the highest probability now.")
print("The model has 'learned' that <bos> is followed by clefG2.")
