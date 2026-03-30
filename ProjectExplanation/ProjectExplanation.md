# Deep-Dive: Sheet Music Transformer (SMT-deep)

> [!NOTE]
> Every claim in this document is grounded in actual source code from this repository.
> No concepts are invented — if something is described here, you can find it in the file cited.

---

## Table of Contents

1. [What Is This Project Solving?](#1-what-is-this-project-solving)
2. [The Big Picture Architecture](#2-the-big-picture-architecture)
3. [The Music Language: bekern](#3-the-music-language-bekern)
4. [File Map](#4-file-map)
5. [Deep Dive: `smt_model/modeling_smt.py`](#5-deep-dive-smt_modelmodeling_smtpy)
6. [Deep Dive: `smt_model/configuration_smt.py`](#6-deep-dive-smt_modelconfiguration_smtpy)
7. [Deep Dive: `data.py`](#7-deep-dive-datapy)
8. [Deep Dive: `smt_trainer.py`](#8-deep-dive-smt_trainerpy)
9. [Deep Dive: `utils.py`](#9-deep-dive-utilspy)
10. [Deep Dive: `eval_functions.py`](#10-deep-dive-eval_functionspy)
11. [Deep Dive: `SynthGenerator.py`](#11-deep-dive-synthgeneratorpy)
12. [Deep Dive: `data_augmentation/`](#12-deep-dive-data_augmentation)
13. [Deep Dive: `ExperimentConfig.py`](#13-deep-dive-experimentconfigpy)
14. [The Three Training Scripts](#14-the-three-training-scripts)
15. [The Full Training Pipeline, Step by Step](#15-the-full-training-pipeline-step-by-step)
16. [How Inference (Prediction) Works](#16-how-inference-prediction-works)
17. [Evaluation Metrics: CER, SER, LER](#17-evaluation-metrics-cer-ser-ler)
18. [Curriculum Learning: The Three Stages](#18-curriculum-learning-the-three-stages)
19. [Glossary](#19-glossary)

---

## 1. What Is This Project Solving?

**Optical Music Recognition (OMR)** is to music what OCR is to text: you take a photograph or scan of printed sheet music and convert it to a machine-readable format.

The hard part: unlike text, music is **polyphonic** — multiple voices/notes can happen *at the same time*. Think of piano music: the left hand and right hand are shown in parallel on the score (the "grand staff"). You can't just read left-to-right like a sentence.

This project solves:
> *Given a full-page image of a piano score, output the complete musical notation as a text sequence in bekern format.*

The approach: treat it as an **image-to-sequence** problem (like translating an image into a sentence). The model is an encoder-decoder Transformer.

---

## 2. The Big Picture Architecture

```
 [Image of score]
       │
       ▼
 ┌───────────────────┐
 │   ConvNeXt CNN    │  ← ENCODER ("The Eyes")
 │  (3 stages)       │    Sees the whole image at once.
 └───────────────────┘    Outputs a 2D grid of feature vectors.
       │
       ▼
 ┌───────────────────┐
 │  2D Positional    │  ← Tells the decoder WHERE each feature came from
 │  Encoding         │    (top-left, bottom-right, etc.)
 └───────────────────┘
       │
       ▼  (flattened to 1D list of features)
 ┌───────────────────┐
 │ Transformer       │  ← DECODER ("The Brain")
 │ Decoder           │    8 layers of self-attention + cross-attention.
 │ (8 layers)        │    Generates one token at a time.
 └───────────────────┘
       │
       ▼
 [Sequence of bekern tokens: <bos> clefG2 <t> clefF4 <b> 4c ... <eos>]
```

---

## 3. The Music Language: bekern

The model does not output musical notes like "C4" directly. It speaks a custom dialect called **bekern**, derived from the Humdrum `**kern` format.

A `**kern` score uses tab-separated columns (one per voice) and newline-separated rows (one per time step). For example:

```
**kern	**kern
*clefG2	*clefF4
*k[f#]	*k[f#]
*M3/4	*M3/4
=1	=1
4c	4r
4d	4E
4e	4F
=2	=2
...
*-	*-
```

Because neural networks need a 1D token stream, whitespace characters are replaced with *special tokens*:

| Real character | Token in bekern |
|---|---|
| space | `<s>` |
| tab | `<t>` |
| newline | `<b>` |

And these *structural* tokens are added:

| Token | Meaning |
|---|---|
| `<bos>` | Begin of sequence (always first) |
| `<eos>` | End of sequence (signals stop) |
| `<pad>` | Padding for batching (index 0)  |

So the token stream for the snippet above would look like:
`<bos> **kern <t> **kern <b> *clefG2 <t> *clefF4 <b> ... <eos>`

This is what the model learns to produce.

---

## 4. File Map

```
SMT-deep/
│
├── smt_model/                    ← The neural network definition
│   ├── __init__.py               ← Exports SMTConfig and SMTModelForCausalLM
│   ├── configuration_smt.py      ← Hyperparameter dataclass
│   ├── modeling_smt.py           ← The actual model (CNN encoder + Transformer decoder)
│   └── architectures/            ← (Experimental alternative architectures, currently empty)
│
├── data.py                       ← Dataset classes & data loading logic
├── smt_trainer.py                ← PyTorch Lightning training wrapper
├── utils.py                      ← Vocabulary building, Levenshtein, kern parsing
├── eval_functions.py             ← Metric computation (CER, SER, LER)
├── SynthGenerator.py             ← Synthetic data generator using Verovio
├── ExperimentConfig.py           ← JSON config file parser (dataclasses)
│
├── data_augmentation/
│   ├── data_augmentation.py      ← augment() and convert_img_to_tensor()
│   └── transforms_custom.py      ← Custom torchvision transforms (Erosion, Dilation, etc.)
│
├── config/                       ← JSON experiment configs (one per experiment)
│   ├── GrandStaff/
│   ├── FP-GrandStaff/
│   ├── FP-Polish_Scores/
│   │   ├── pretraining.json      ← Stage 1 config (synthetic system-level data)
│   │   └── finetuning.json       ← Stage 2 config (real full-page data)
│   └── ...
│
├── train.py                      ← Entry point: standard system-level training
├── fp-train-1.py                 ← Entry point: full-page pretraining (synthetic)
├── fp-train-2.py                 ← Entry point: full-page curriculum learning (real data)
│
├── vocab/                        ← Saved vocabulary files (.npy dicts)
├── weights/                      ← Saved model checkpoints (.ckpt files)
└── Generator/
    └── paper_textures/           ← Texture images for synthetic data realism
```

---

## 5. Deep Dive: `smt_model/modeling_smt.py`

This is the most important file. Let's go class by class.

---

### `PositionalEncoding2D`

```python
class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, h_max, w_max): ...
    def forward(self, x): ...
```

**Problem it solves:** A CNN gives us a feature grid (height × width). But when we later flatten this grid into a 1D list of features to feed into the Transformer, we lose the information of *where* each feature was spatially.

**How it works:** It pre-computes a sinusoidal position table of shape `(dim, h_max, w_max)`. The height dimension uses sine/cosine in the first half of `dim`, the width dimension uses them in the second half. When you call `forward(x)`, it simply *adds* this encoding to the feature map — so the encoder's output now carries both "what is here" (from the CNN) and "where is here" (from the positional encoding).

---

### `PositionalEncoding1D`

```python
class PositionalEncoding1D(nn.Module):
    def __init__(self, dim, len_max): ...
    def forward(self, x, start=0): ...
```

Same idea, but for the decoder's **text sequence**. The decoder gets a sequence of token embeddings (like word embeddings). Without position info, it can't tell which token came first. This adds a sinusoidal encoding to each token embedding based on its *index* in the sequence.

The `start` parameter is used during inference: when we generate token by token, we can offset the position encoding so token #5 gets the correct position.

---

### `MultiHeadAttention`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, bias=True): ...
    def forward(self, query, key=None, value=None, ...): ...
```

**The heart of the Transformer.** If you've studied Transformers, you know attention. Here's a quick re-fresher:

The attention mechanism answers: *"Given my current state (query), which parts of the context (keys/values) are most relevant?"*

- It projects `query`, `key`, `value` through separate linear layers.
- Then splits them into `num_heads` parallel "heads" — each head can attend to different things.
- Computes scores: `scores = (Q @ K^T) / sqrt(d_head)`.
- Softmax the scores → attention weights (probabilities summing to 1).
- Weighted sum of values: `output = attention_weights @ V`.

**Key implementation detail — Flash Attention:** The code checks `hasattr(F, 'scaled_dot_product_attention')`. On modern PyTorch + GPU, this uses the Flash Attention kernel which is much faster and more memory-efficient. It's used automatically if available.

**Two modes of use:**
- `key=None, value=None` → **Self-attention**: query, key, value all come from the same sequence (the decoder attending to itself).
- `key` and `value` provided → **Cross-attention**: query from the decoder, key/value from the encoder (decoder attending to the image features).

---

### `DecoderLayer`

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1, activation="relu"): ...
    def forward(self, x, encoder_output_key, encoder_output_value, tgt_mask=None, ...): ...
```

One full decoder block, stacked 8 times. Each layer does three things in sequence:

1. **Self-Attention:** The token sequence attends to *itself*. This is how the model builds up context about what it has already generated. A **causal mask** (`tgt_mask`) ensures that token position `i` can only see positions `0..i` (no "peeking into the future").

2. **Cross-Attention:** The token sequence queries the **encoder output** (image features). `encoder_output_key` gets the 2D position-encoded features (so the decoder knows *where* it's looking), while `encoder_output_value` gets the raw features (the actual visual content). This asymmetry is intentional — the keys carry spatial context while the values carry semantic content.

3. **Feed-Forward Network (FFN):** Two linear layers with a ReLU/GELU activation. This is where the model does non-linear reasoning on each token independently.

Each sub-operation has **Residual connections** (`x = x + output`) and **LayerNorm** — standard Transformer practice to prevent gradients from vanishing.

---

### `DecoderStack`

```python
class DecoderStack(nn.Module):
    def __init__(self, num_dec_layers, d_model, dim_ff, num_heads, dropout): ...
    def forward(self, x, encoder_output_2D, encoder_output_raw, ...): ...
```

Simply stacks `num_dec_layers` (8 by default) `DecoderLayer` instances and calls them in order. The output of layer `i` becomes the input to layer `i+1`.

---

### `Decoder`

```python
class Decoder(nn.Module):
    def __init__(self, num_dec_layers, d_model, dim_ff, n_heads, max_seq_length, out_categories, ...): ...
    def forward(self, decoder_input, encoder_output_2D, encoder_output_raw, ...): ...
```

The full decoder module. Adds three things on top of `DecoderStack`:

1. **Token Embedding** (`nn.Embedding`): Converts integer token indices into dense vectors of size `d_model` (256). Think of this as a lookup table: token index 42 → a learned 256-dimensional vector.
2. **1D Positional Encoding**: Added to the embeddings.
3. **Vocabulary Projection** (`nn.Linear(d_model, out_categories)`): At the end, maps the 256-dim output of each token position back to a score for every symbol in the vocabulary. The highest score = the predicted token.

---

### `SMTModelForCausalLM` — The Main Model

```python
class SMTModelForCausalLM(PreTrainedModel):
    def __init__(self, config: SMTConfig): ...
    def forward_encoder(self, x): ...
    def forward_decoder(self, encoder_output, last_predictions, ...): ...
    def forward(self, encoder_input, decoder_input, labels=None): ...
    def predict(self, input, convert_to_str=False, ...): ...
```

This is the glue that ties the encoder and decoder together. It inherits from HuggingFace's `PreTrainedModel`, which gives it `.from_pretrained()` and `.save_pretrained()` for free.

**`__init__`:**
- Creates a `ConvNextModel` (from HuggingFace Transformers) with 3 stages and `[64, 128, 256]` channels.
- The total spatial downsampling factor is `2^(3+1) = 16`. So a 512×2048 image → 32×128 feature grid.
- Creates the `Decoder`.
- Creates `PositionalEncoding2D` sized to the downscaled feature grid.
- Creates `CrossEntropyLoss` with `ignore_index=padding_token` (so padding tokens don't contribute to the loss).
- Stores `w2i` (word-to-index) and `i2w` (index-to-word) dictionaries.

**`forward_encoder(x)`:**
Feeds the image through ConvNeXt and returns the `last_hidden_state` — a tensor of shape `(B, 256, H/16, W/16)`.

**`forward_decoder(encoder_output, last_predictions)`:**
1. Add 2D positional encoding to the encoder output.
2. Flatten the 2D spatial grid into a 1D sequence: `(B, C, H, W) → (B, H*W, C)`. This is now a "list of image patches."
3. Generate two masks:
   - **Token padding mask** (`_generate_token_mask`): A boolean mask of shape `(B, SeqLen)`. `True` where real tokens are, `False` where padding is. This prevents the model from attending to padding.
   - **Causal mask** (`_generate_causal_mask`): An upper-triangular boolean matrix of shape `(SeqLen, SeqLen)`. Position `i,j` is `True` if `j > i`, meaning "position `i` cannot see future position `j`." This is what makes autoregressive generation possible.
4. Calls the decoder.

**`forward(encoder_input, decoder_input, labels=None)`:**
The training forward pass. Runs encoder, then decoder. If `labels` are provided, computes Cross-Entropy loss.

**`predict(input, ...)`:**
Autoregressive inference. Starts with `[<bos>]`, runs the decoder, picks the highest-scoring next token (argmax), appends it to the sequence, and repeats until `<eos>` is produced or `maxlen` is reached. Note that it re-runs the *full decoder* on the growing sequence at every step — there is no key-value caching implemented here.

---

## 6. Deep Dive: `smt_model/configuration_smt.py`

```python
class SMTConfig(PretrainedConfig):
    def __init__(self, maxh=3508, maxw=2480, maxlen=1512, out_categories=2512, ...):
```

This is just a **Python dataclass** describing all hyperparameters. Inheriting from `PretrainedConfig` lets it be serialized/deserialized as JSON when saving/loading model weights.

Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `maxh` | 3508 | Max image height the model was built for |
| `maxw` | 2480 | Max image width |
| `maxlen` | 1512 | Max sequence length the decoder can generate |
| `out_categories` | 2512 | Vocabulary size |
| `d_model` | 256 | Embedding dimension throughout the model |
| `dim_ff` | 256 | Hidden size of the FFN in each decoder layer |
| `num_dec_layers` | 8 | Number of stacked decoder layers |
| `attn_heads` | 4 | Number of attention heads per layer |
| `w2i` | `{}` | Word-to-index dict (filled at training time) |
| `i2w` | `{}` | Index-to-word dict |

---

## 7. Deep Dive: `data.py`

This file contains all the Dataset and DataModule classes. Understanding the class hierarchy is key.

### Class Hierarchy

```
torch.utils.data.Dataset
  └── OMRIMG2SEQDataset          (base: teacher forcing, tokenization, augment flag)
        ├── GrandStaffSingleSystem   (loads from HuggingFace, single systems)
        │     └── GrandStaffFullPage     (loads full-page data)
        │           └── CurriculumTrainingDataset (adds VerovioGenerator + staged mixing)
        │                 └── GrandStaffFullPageCurriculumLearning
        └── SyntheticOMRDataset        (100% generated by VerovioGenerator)

lightning.LightningDataModule
  ├── GrandStaffDataset              (wraps GrandStaffSingleSystem for real data training)
  ├── SyntheticGrandStaffDataset     (wraps SyntheticOMRDataset for fp-train-1.py)
  └── SyntheticCLGrandStaffDataset   (wraps GrandStaffFullPageCurriculumLearning for fp-train-2.py)
```

---

### `prepare_data(sample, reduce_ratio, fixed_size)` — For Single-System Data

```python
def prepare_data(sample, reduce_ratio=1.0, fixed_size=None):
```

A HuggingFace `.map()` function applied to each raw dataset sample. It:

1. Converts the `PIL.Image` to a NumPy array.
2. Resizes it. If the image is wider than 3056px, it caps the width at `3056 * reduce_ratio`. Otherwise, it scales proportionally.
3. Processes the text transcription:
   - Strips whitespace/newlines.
   - Removes bar numbers after `=` signs (e.g., `=1` → `=`).
   - Replaces space, tab, newline with their special tokens.
   - Wraps with `<bos>` ... `<eos>`.

---

### `prepare_fp_data(sample, reduce_ratio, krn_format)` — For Full-Page Data

```python
def prepare_fp_data(sample, reduce_ratio, krn_format):
```

Same idea, but calls `parse_kern()` from `utils.py` (which does a full bekern normalization including removing forbidden tokens like `*staff1`, `*staff2`, `*ped` etc., using the `clean_kern` function).

---

### `batch_preparation_img2seq(data)` — The Collate Function

```python
def batch_preparation_img2seq(data):
```

This is the `collate_fn` argument passed to `DataLoader`. When PyTorch assembles a batch from individual samples, all images have different sizes and all sequences have different lengths. This function:

1. **Pads images:** Creates a white (all-ones) tensor of shape `(B, 1, max_H, max_W)` and copies each image into the top-left corner. White pixels (value 1.0) correspond to the blank areas of the padding — neutral input.
2. **Pads sequences:** Creates zero tensors (padding index 0) and fills in the actual token indices.
3. Returns `(X_train, decoder_input, y)` where:
   - `X_train`: batch of padded images.
   - `decoder_input`: the ground truth sequence *shifted right* — it starts at `<bos>` and ends one before `<eos>`. This is what the decoder receives as input during training.
   - `y`: the ground truth sequence *shifted left* — it starts one after `<bos>` and ends at `<eos>`. This is what the decoder must predict.

**Why the shift?** This is called **Teacher Forcing**. At training step `t`, the decoder sees all correct tokens `0..t-1` as input, and must predict token `t`. This makes training stable because the model doesn't have to recover from its own early mistakes.

---

### `OMRIMG2SEQDataset.apply_teacher_forcing(sequence)`

```python
def apply_teacher_forcing(self, sequence):
    errored_sequence = sequence.clone()
    for token in range(1, len(sequence)):
        if np.random.rand() < self.teacher_forcing_error_rate and sequence[token] != self.padding_token:
            errored_sequence[token] = np.random.randint(0, len(self.w2i))
    return errored_sequence
```

This adds **noise** to the ground-truth input sequence. With probability `teacher_forcing_error_rate` (20% by default), a token is replaced with a random token from the vocabulary. This is a regularization technique: the model learns to recover from bad input tokens instead of always relying on perfect context. It makes the model more robust at inference time.

---

### `CurriculumTrainingDataset.__getitem__(index)`

```python
def __getitem__(self, index):
    step = self.trainer.global_step
    stage = (step // self.increase_steps) + self.curriculum_stage_beginning
    ...
```

This is the most complex dataset class. On every call to `__getitem__`, it:

1. Checks the current global training step from the trainer.
2. Calculates which **curriculum stage** we're in (`stage`).
3. If still in the "building up" phases (`stage < num_cl_steps + start`): generates a **synthetic** score with `random.randint(1, stage)` systems — so stage 2 can have 1-2 systems, stage 3 can have 1-3, etc. This is the curriculum: gradually increasing complexity.
4. If in the "fine-tuning" phase: mixes synthetic and real data according to a **linearly decaying probability** — starting at 90% synthetic, slowly going to 20% synthetic. This gradually transitions the model from synthetic to real data.

---

## 8. Deep Dive: `smt_trainer.py`

```python
class SMT_Trainer(L.LightningModule):
```

**PyTorch Lightning** abstracts away the training loop. Instead of writing:
```python
for epoch in ...:
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```
You just define methods and Lightning handles the loop.

---

### `__init__`

Creates the `SMTConfig` and `SMTModelForCausalLM`. Also calls `summary()` from `torchinfo` to print a table of layer names, output shapes, and parameter counts — very useful for debugging.

---

### `configure_optimizers()`

```python
return torch.optim.Adam(
    list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
    lr=1e-4, amsgrad=False
)
```

Uses the **Adam optimizer** with learning rate `1e-4`. Adam is an adaptive gradient method — it keeps a running estimate of both the first moment (mean) and second moment (variance) of gradients, which lets it effectively have a per-parameter learning rate. It converges much faster than plain SGD for sequence models.

Note: only the `encoder` and `decoder` parameters are optimized — not `pos2D`, which is a fixed buffer.

---

### `training_step(batch)`

```python
def training_step(self, batch):
    x, di, y = batch
    outputs = self.model(encoder_input=x, decoder_input=di, labels=y)
    loss = outputs.loss
    ...
    return loss
```

The one function Lightning calls every iteration. It unpacks `(image, decoder_input, labels)`, runs a forward pass, and returns the loss. Lightning automatically calls `.backward()` and `optimizer.step()` after this returns.

---

### `validation_step(val_batch)` and `on_validation_epoch_end()`

During validation, we don't use teacher forcing at all. We call `model.predict(input=x)` which runs the model autoregressively — exactly as it would at inference/deployment time. The predictions and ground truths are accumulated and then CER/SER/LER are computed at the end of each validation epoch.

---

## 9. Deep Dive: `utils.py`

### `levenshtein(a, b)`
Classic dynamic-programming implementation of edit distance. Used for all three metrics. Time complexity: `O(|a| × |b|)`.

### `check_and_retrieveVocabulary(YSequences, pathOfSequences, nameOfVoc)`
Looks for pre-saved vocabulary `.npy` files. If found, loads them. If not, calls `make_vocabulary()` to build them from scratch and saves them. This is how the vocabulary is persistent across runs.

### `make_vocabulary(YSequences, pathToSave, nameOfVoc)`
Iterates over all training transcriptions, collects all unique tokens into a Python `set`, and assigns each a unique integer index (starting at 1). Index 0 is reserved for `<pad>`. Returns two dicts: `w2i` (token string → int) and `i2w` (int → token string).

### `parse_kern(krn, krn_format="bekern")`
The main text pre-processing function. Takes a raw kern string and:
1. Calls `clean_kern()` to remove forbidden structural tokens.
2. Strips bar numbers after `=`.
3. Replaces space/tab/newline with `<s>/<t>/<b>`.
4. Handles `·` and `@` characters differently depending on the `krn_format`:
   - `kern`: removes them both.
   - `ekern`: `·` → space, `@` removed.
   - `bekern`: both `·` → space, `@` → space. This longest format captures more notational detail.

---

## 10. Deep Dive: `eval_functions.py`

### `compute_poliphony_metrics(hyp_array, gt_array)`

The main evaluation function. Takes two lists of strings (model hypothesis and ground truth), and computes three metrics — at three different granularity levels:

| Metric | Granularity | Split on |
|---|---|---|
| **CER** (Character Error Rate) | Character | `list(string)` — every single character |
| **SER** (Symbol Error Rate) | Token/Symbol | Split on spaces, `<b>`, `<t>` |
| **LER** (Line Error Rate) | Line | Split on `\n` |

All three use `levenshtein(hypothesis, ground_truth) / len(ground_truth) * 100`.

Lower is better. The paper reports SER as the main metric.

---

## 11. Deep Dive: `SynthGenerator.py`

This file is critical for understanding how training data is generated. There's no real labeled music data on the scale needed — so we *render* it.

### `VerovioGenerator.__init__(sources, split, krn_format)`

- `sources`: A HuggingFace dataset reference (e.g., `"antoniorv6/grandstaff-ekern"`) or local path.
- Loads all transcriptions and organizes them into `self.beat_db`: a dictionary keyed by **time signature** (e.g., `*M3/4`, `*M4/4`), where each key maps to a list of systems in that meter. This grouping ensures that when we concatenate multiple systems on a page, they at least share the same time signature (otherwise the rhythmic rendering would be broken).
- Initializes `verovio.toolkit()` — Verovio is a music engraving library. It takes kern format as input and renders it as SVG/PNG.
- Initializes a random title/author generator (`RandomSentence`, `names`).
- Loads paper texture images from `Generator/paper_textures/`.

### `generate_music_system_image(reduce_ratio=0.5)`

1. Randomly picks a time signature and a music sequence in that meter.
2. Reconstructs the kern string from bekern tokens.
3. Renders it with Verovio (random visual parameters: bar line width, staff line width, spacing).
4. Checks that the result is exactly **one system** (one grand staff) using `count_class_occurrences(class_name='grpSym')`. Retries until this is true.
5. Converts SVG → PNG, crops to the content, resizes.
6. Returns the image and the bekern token sequence wrapped in `<bos>...<eos>`.

### `generate_full_page_score(max_systems, ...)`

1. Picks `max_systems` music sequences with the same time signature.
2. Concatenates them into a single kern "page" using `filter_system_continuation()` (strips the clef/key/meter from the beginning of each continuation system — they're redundant after the first).
3. Optionally adds a random title and author name at the top.
4. Renders the full page with Verovio.
5. **Inkifies** the image: applies an `oil_paint` effect to simulate handwriting or old printing.
6. **Composites with paper texture**: takes a random crop from one of the paper texture images and blends the music onto it, making it look like real aged sheet music.
7. Returns image + bekern sequence.

---

## 12. Deep Dive: `data_augmentation/`

### `augment(image)` in `data_augmentation.py`

Applied to training images (never validation or test). It randomly applies a chain of transforms — each with independent probability 0.2:

| Transform | Effect |
|---|---|
| `RandomPerspective` | Simulates a tilted camera angle |
| `ElasticDistortion` | Warps the image as if the paper is slightly crumpled |
| `RandomTransform` | Random affine transformation |
| `Erosion` or `Dilation` | Makes strokes thinner (erode) or thicker (dilate) — simulates print quality |
| `BrightnessAdjust` | Randomly changes brightness |
| `ContrastAdjust` | Randomly changes contrast |
| `Grayscale` | Converts to single-channel (always applied) |
| `ToTensor` | Converts to float tensor in `[0,1]` (always applied) |

### `convert_img_to_tensor(image)` 
Used for validation/test: just Grayscale + ToTensor, no random transforms.

---

## 13. Deep Dive: `ExperimentConfig.py`

Parses the `.json` config files using Python dataclasses.

```json
{
    "data": {
        "data_path": "antoniorv6/grandstaff-ekern",
        "batch_size": 1,
        "vocab_name": "Polish_Scores_BeKern",
        "num_workers": 20,
        "krn_format": "bekern",
        "reduce_ratio": 0.5
    },
    "checkpoint": {
        "dirpath": "weights/Polish_Scores",
        "filename": "FP-Polish_Scores-system-level",
        "monitor": "val_SER"
    }
}
```

- `data_path`: HuggingFace dataset ID or local path.
- `vocab_name`: Determines the filename of the saved vocabulary in `vocab/`.
- `reduce_ratio`: Images are downscaled by this factor (0.5 = half size). This saves GPU memory for large full-page images.
- `monitor`: The metric that determines which checkpoint is "the best" (`val_SER` = Symbol Error Rate on validation set).

---

## 14. The Three Training Scripts

| Script | Dataset Class Used | Purpose |
|---|---|---|
| `train.py` | `GrandStaffDataset` | Standard training on real single-system data |
| `fp-train-1.py` | `SyntheticGrandStaffDataset` | Full-page pretraining on **synthetic** system-level data only |
| `fp-train-2.py` | `SyntheticCLGrandStaffDataset` | Full-page fine-tuning with **curriculum learning** — load the stage-1 checkpoint and continue |

All three scripts follow the same pattern:
1. Parse JSON config → `ExperimentConfig`.
2. Create the appropriate `LightningDataModule`.
3. Create (or load) `SMT_Trainer`.
4. Set up `WandbLogger`, `ModelCheckpoint`, `EarlyStopping` callbacks.
5. Call `trainer.fit()`.
6. Load the best checkpoint and call `trainer.test()`.

**`fp-train-2.py` additionally:**
- Takes a `--starting_checkpoint` argument to load the pretrained weights.
- Sets up a second `ModelCheckpoint` that saves a separate checkpoint at each curriculum stage transition (`stage_checkpointer`). This lets you resume from any curriculum stage.
- Sets `min_steps=300000` — training runs for at least 300k steps regardless of early stopping.

---

## 15. The Full Training Pipeline, Step by Step

Here's what actually happens when you run `uv run fp-train-1.py --config_path config/FP-Polish_Scores/pretraining.json`:

1. **Config parsing:** JSON → `ExperimentConfig` object.
2. **Data loading:** `SyntheticGrandStaffDataset` is created.
   - `VerovioGenerator` loads the HuggingFace dataset (`antoniorv6/grandstaff-ekern`).
   - Organizes sequences by time signature into `beat_db`.
3. **Vocabulary:** `check_and_retrieveVocabulary()` is called with the GT sequences from train/val/test splits. Builds `w2i` and `i2w` and saves them as `vocab/Polish_Scores_BeKernw2i.npy`.
4. **Model creation:** `SMTModelForCausalLM` is built with the vocab size and max dimensions.
5. **Training loop (Lightning):**
   - For each batch: `DataLoader` draws a random index → `SyntheticOMRDataset.__getitem__()` → calls `generator.generate_music_system_image()` → Verovio renders an image on the fly → augmentation → tokenization → returns `(image, decoder_input, labels)`.
   - `batch_preparation_img2seq` pads all images and sequences in the batch.
   - `SMT_Trainer.training_step()`: forward pass → Cross-Entropy loss → loss logged to wandb.
   - PyTorch Lightning: `.backward()` → `optimizer.step()` → `optimizer.zero_grad()`.
6. **Every 5 epochs:** Validation loop runs. `model.predict()` is called autoregressively on validation images. CER/SER/LER are computed and logged.
7. **Checkpointing:** The checkpoint with lowest `val_SER` is saved to `weights/Polish_Scores/FP-Polish_Scores-system-level.ckpt`.
8. **Early Stopping:** If `val_SER` doesn't improve by more than 0.01 for 5 consecutive validation epochs, training stops.
9. **Testing:** The best checkpoint is loaded and evaluated on the test split.

---

## 16. How Inference (Prediction) Works

```python
predictions, _ = model.predict(convert_img_to_tensor(image).unsqueeze(0).to(device),
                               convert_to_str=True)
```

Step by step inside `SMTModelForCausalLM.predict()`:

1. `predicted_sequence = [w2i['<bos>']]` — start the sequence.
2. Run `forward_encoder(input)` once to get `encoder_output`. Store it.
3. **Loop:**
   - Run `forward_decoder(encoder_output, predicted_sequence)`.
   - Get `output.logits[:, -1, :]` — the scores for the *last* position (the next token).
   - `argmax()` → pick the highest-scoring token.
   - Append to `predicted_sequence`.
   - If token is `<eos>` → stop.
   - If we've reached `maxlen` → stop.
4. Convert indices back to strings using `i2w`.
5. Replace `<s>` → ` `, `<t>` → `\t`, `<b>` → `\n` to get the human-readable kern output.

---

## 17. Evaluation Metrics: CER, SER, LER

All three are **edit distance rates** — the lower, the better.

**CER (Character Error Rate):**
The hypothesis and ground truth strings are compared character by character. Even a single extra space counts as an error. Very sensitive, very strict.

**SER (Symbol Error Rate):**
The strings are split into tokens (musical symbols). Something like `4c` (a quarter note C) is one token. This is the **main metric** reported in the paper. It measures how many notes/events the model got wrong.

**LER (Line Error Rate):**
The strings are split into lines (one time step per line). An entire measure line counts as one unit. Measures how many "rows" of the score are wrong.

---

## 18. Curriculum Learning: The Three Stages

In `CurriculumTrainingDataset`:

```python
self.increase_steps: int = 40000
self.num_cl_steps: int = 3
# stage = (global_step // 40000) + 2
```

| Step Range | Stage | What the model sees |
|---|---|---|
| 0 – 39,999 | 2 | Synthetic 1-2 systems per "page" |
| 40,000 – 79,999 | 3 | Synthetic 1-3 systems per "page" |
| 80,000 – 119,999 | 4 | Synthetic 1-4 systems per "page" |
| 120,000+ | Fine-tuning | Mix of synthetic (90% → 20%) and real full-page data |

The synthetic probability decays linearly from 90% to 20% as training progresses (`linear_scheduler_synthetic`), gradually forcing the model to handle real scanned music.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **Token** | A single atomic unit in the vocabulary (one musical symbol, or a structural marker like `<b>`) |
| **Embedding** | A learned dense vector representation of a token |
| **d_model** | The dimensionality of all embeddings and hidden states in the model (256 here) |
| **Attention head** | One parallel unit of the multi-head attention mechanism |
| **Causal mask** | An upper-triangular mask preventing the decoder from seeing future tokens |
| **Teacher forcing** | Training trick: feed ground truth tokens as decoder input instead of the model's own predictions |
| **Cross-entropy loss** | The training objective. Measures "surprise" at the predicted token distribution vs. the true token |
| **Gradient** | The derivative of the loss with respect to a weight — tells us which direction to nudge the weight |
| **Adam optimizer** | An adaptive gradient optimizer that adjusts learning rates per parameter |
| **Curriculum learning** | Training strategy: start with simpler examples, gradually increase complexity |
| **bekern** | The text format used for music transcription: kern notation with whitespace replaced by special tokens |
| **ConvNeXt** | A modern CNN architecture. Used here as the image encoder |
| **Verovio** | A music engraving library. Converts kern text → SVG/PNG images |
| **SER** | Symbol Error Rate — main evaluation metric (lower = better) |
| **Lightning** | PyTorch Lightning: a framework abstracting the training loop boilerplate |
