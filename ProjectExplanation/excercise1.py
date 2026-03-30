"""
EXERCISE 1: Implement the Training Loop from Scratch

★ GOAL: Understand what happens INSIDE every training iteration of the SMT.
        You will build the exact same mechanism that smt_trainer.py uses,
        but WITHOUT any abstractions (no Lightning, no wandb).

★ YOUR TASK: Fill in the 4 marked sections (STEP 1 through STEP 4).
             Each section has a comment explaining what you need to do.

★ EXPECTED RESULT when correct:
        Iter    1 |  Loss: ~1.60  |  Predicted: (random)
        Iter    5 |  Loss: ~0.90  |  Predicted: (improving)
        Iter   30 |  Loss: ~0.05  |  Predicted: 4c  ✓

★ HOW TO CHECK: At iter 30, the model should predict '4c' for every context.
                The loss should clearly go down iteration by iteration.

Run with: python excercise1.py

────────────────────────────────────────────────────────────────────────────────
BACKGROUND — how SMT's training_step works:

    def training_step(self, batch):
        x, di, y = batch                                   # unpack batch
        outputs = self.model(encoder_input=x,              # FORWARD PASS
                             decoder_input=di, labels=y)
        loss = outputs.loss                                # LOSS
        ...
        return loss                                        # Lightning does backward + step

The loss is CrossEntropyLoss between:
    - outputs.logits:  shape (Batch, SeqLen, VocabSize) — model's predictions
    - y:               shape (Batch, SeqLen)             — correct token indices

CrossEntropyLoss(logits.permute(0,2,1), y)   ← note the permute! PyTorch CE
                                                 expects (Batch, Classes, SeqLen)

────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
# Setup (do NOT modify)
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)

VOCAB = {0: "<pad>", 1: "<bos>", 2: "clefG2", 3: "4c", 4: "<eos>"}
W2I   = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)

# A tiny sequence model (token-level, no images — just text prediction)
class MiniTranscriber(nn.Module):
    """
    Given a context of token indices, predict the next token at each position.
    This is the decoder-only equivalent of what SMT's Decoder does.
    """
    def __init__(self, vocab_size, d_model=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.rnn   = nn.GRU(d_model, d_model, batch_first=True)
        self.proj  = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (Batch, SeqLen) of token indices
        emb  = self.embed(x)             # (Batch, SeqLen, d_model)
        out, _ = self.rnn(emb)           # (Batch, SeqLen, d_model)
        return self.proj(out)            # (Batch, SeqLen, VocabSize) — logits


model = MiniTranscriber(VOCAB_SIZE)

# Training data: teach the model to output this sequence
# Given: <bos>  clefG2  <t>    4c
# Predict: clefG2  <t>    4c   <eos>
decoder_input = torch.tensor([[W2I["<bos>"], W2I["clefG2"], W2I["4c"], W2I["4c"]]])
labels        = torch.tensor([[W2I["clefG2"], W2I["4c"],    W2I["4c"], W2I["<eos>"]]])

# ─────────────────────────────────────────────────────────────────────────────
# YOUR CODE STARTS HERE
# ─────────────────────────────────────────────────────────────────────────────

# ── Configure the optimizer ──────────────────────────────────────────────────
# HINT: The SMT uses torch.optim.Adam with lr=1e-4.
#       Create an optimizer here that will optimize model.parameters().
#       Try lr=0.05 for faster convergence in this toy example.
#
# optimizer = ???    ← FILL THIS IN

# ── Configure the loss function ──────────────────────────────────────────────
# HINT: The SMT uses nn.CrossEntropyLoss(ignore_index=0)
#       (ignore_index=0 means padding tokens don't contribute to the loss)
#
# loss_fn = ???      ← FILL THIS IN

print("=" * 60)
print("Training the MiniTranscriber")
print(f"{'Iter':>5}  {'Loss':>8}  {'Prediction at pos 0':>22}")
print("-" * 60)

for i in range(30):

    # ── STEP 1: FORWARD PASS ────────────────────────────────────────────────
    # Run the model on decoder_input.
    # Store the result in a variable called 'logits'.
    # Expected shape: (1, 4, VOCAB_SIZE)
    #
    # logits = ???   ← FILL THIS IN

    # ── STEP 2: COMPUTE LOSS ────────────────────────────────────────────────
    # CrossEntropyLoss expects: (Batch, VocabSize, SeqLen) vs (Batch, SeqLen)
    # You need to permute logits from (B, S, V) → (B, V, S) before passing it.
    # HINT: logits.permute(0, 2, 1)
    #
    # loss = ???     ← FILL THIS IN

    # ── STEP 3: BACKWARD PASS ───────────────────────────────────────────────
    # Clear old gradients, then compute new ones.
    # HINT: Two lines — optimizer.zero_grad() and loss.backward()
    #
    # ???            ← FILL THIS IN

    # ── STEP 4: OPTIMIZER STEP ──────────────────────────────────────────────
    # Tell the optimizer to update the weights.
    # HINT: One line — optimizer.step()
    #
    # ???            ← FILL THIS IN

    # Logging (do not modify)
    with torch.no_grad():
        pred_idx  = logits[0, 0].argmax().item()
        pred_word = VOCAB[pred_idx]
    print(f"{i+1:>5}  {loss.item():>8.4f}  {pred_word:>22}")

print()
print("If the loss reached below 0.1 and the prediction is 'clefG2',")
print("you have correctly implemented the training loop!")

# ─────────────────────────────────────────────────────────────────────────────
# BONUS QUESTION (think about it, no code needed):
#
# After 30 iterations on ONE example, the model might have overfit perfectly.
# What would happen if you now fed it a DIFFERENT starting token, e.g. <eos>?
# Would it still output "clefG2"? Why or why not?
# (This is why we train on thousands of examples, not just one.)
# ─────────────────────────────────────────────────────────────────────────────
