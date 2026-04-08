# Music Notation Formats

Before a neural network can learn to read sheet music, the visual content of a score must be expressed as a target text sequence — the "language" the model is trained to produce. The choice of this target representation is far from trivial. It determines the size of the model's vocabulary, the length of its output sequences, and the degree to which the textual tokens correlate with identifiable graphical elements in the image. A representation that is too coarse collapses musical detail; one that is too fine explodes the vocabulary into tens of thousands of rarely-seen symbols that are difficult to learn. This chapter introduces the standard digital music formats, explains why they are unsuitable for end-to-end OMR, and then describes the Humdrum `**kern` encoding and its tokenisation variants that are used as the target format in this work.

## Standard Digital Music Formats

Three major families of digital music representation exist today. **MIDI**, introduced in 1983, is a performance protocol that encodes which notes to play, at what velocity, and for how long — but discards notational semantics entirely. It cannot distinguish between a dotted quarter note and a tied quarter-plus-eighth, collapses enharmonic spellings (C♯ and D♭ share the same pitch number), and contains no information about staff layout, clefs, or beaming. **MusicXML** is an XML-based interchange format that does encode full notational detail — pitch, duration, voice assignment, clefs, key signatures, articulations — but its extreme verbosity (a single quarter note C4 produces over a dozen XML tags) makes it impractical for autoregressive sequence generation. **MEI** (Music Encoding Initiative) extends MusicXML's approach with even richer scholarly metadata (critical apparatus, variant readings, provenance), further increasing document length.

None of these formats provides the combination of properties needed for end-to-end OMR: compact token sequences, unambiguous serialisation, preservation of musical semantics, and a manageable vocabulary size. The Humdrum `**kern` encoding, described in the following section, fills exactly this niche.

## The Humdrum **kern Encoding

### Origins and Design Philosophy

Humdrum is a set of command-line tools and data representations for music analysis, developed by David Huron at Ohio State University in the 1990s. Its core data representation, `**kern`, was designed for *computational musicology* — systematic, quantitative analysis of musical corpora. This analytical orientation gives `**kern` two properties that make it well-suited for OMR:

1. **Compactness.** A single `**kern` token like `8e-JL` encodes duration (`8` = eighth note), pitch (`e`), accidental (`-` = flat), beam end (`J`), and beam start (`L`) in just five characters. The same information requires over a dozen XML tags in MusicXML.
2. **Tabular structure.** Polyphonic music is represented as a grid of tab-separated columns (called *spines*) and newline-separated rows (called *records*). Each column corresponds to a musical voice; each row corresponds to a simultaneous time slice. This structure maps naturally onto the two-dimensional layout of a musical score.

### Pitch Encoding

Pitch in `**kern` uses an absolute letter-name system (A–G) combined with a case-and-repetition scheme for octave designation:

| Kern token | Pitch | Octave |
|---|---|---|
| `CCC` | C | 1 |
| `CC` | C | 2 |
| `C` | C | 3 |
| `c` | C | 4 (middle C) |
| `cc` | C | 5 |
| `ccc` | C | 6 |

Accidentals are appended directly after the letter: `#` for sharp, `-` for flat, `n` for natural, with repetition for double accidentals (`##`, `--`). All pitches are encoded at concert pitch regardless of transposing instruments.

### Rhythm Encoding

Duration uses a *reciprocal* system: the number indicates how many notes of that duration fit into a whole note.

| Number | Duration |
|---|---|
| `0` | Breve (2 whole notes) |
| `1` | Whole note |
| `2` | Half note |
| `4` | Quarter note |
| `8` | Eighth note |
| `16` | Sixteenth note |

Dotted rhythms append a period: `4.` is a dotted quarter note (duration = 1/4 + 1/8 = 3/8 of a whole note). Tuplets are encoded by the reciprocal of their actual duration: a quarter-note triplet (three notes in the space of two quarter notes, or equivalently 12 divisions of a whole note) is written as `12`.

### Structural Tokens

Beyond pitch and rhythm, `**kern` files contain interpretive records (prefixed with `*`) and barlines (prefixed with `=`):

- `*clefG2` — treble clef on the second staff line
- `*clefF4` — bass clef on the fourth staff line
- `*k[f#c#]` — key signature with F♯ and C♯ (D major / B minor)
- `*M3/4` — time signature of 3/4
- `=` — barline (optionally followed by a bar number, e.g. `=15`)
- `*-` — spine terminator (end of that voice)

### Polyphonic Representation: The Spine Grid

A two-voice piano passage in `**kern` looks like this:

```
**kern      **kern
*clefG2     *clefF4
*k[f#]      *k[f#]
*M3/4       *M3/4
=1          =1
4c          4E
4d          4F#
4e          4G
=2          =2
2.g         2.B
=3          =3
*-          *-
```

The left column is the upper staff (treble clef), the right column is the lower staff (bass clef), and columns are separated by tab characters. Each row is a synchronous time-slice: when the upper voice plays `4c` (quarter note C4), the lower voice simultaneously plays `4E` (quarter note E3). The `*-` tokens terminate both spines at the end.

Within a single spine, multiple simultaneous notes of the same duration (chords) are separated by spaces: `4c 4e 4g` represents a C major chord of quarter notes.

This tabular structure is the key advantage of `**kern` for polyphonic OMR: it preserves the temporal alignment of voices that is visible in the score image, while remaining a flat text format that can be serialised into a 1D token stream.

## Agnostic vs. Semantic Representations

In the broader OMR literature, output representations fall into two fundamentally different categories.

### Agnostic Representation

An agnostic representation describes the *visual appearance* of the score without interpreting its musical meaning. Tokens describe graphical primitives and their spatial positions on the staff:

```
notehead-L3  stem-up  flag-8th  notehead-S2  accidental-sharp  ...
```

Here, `notehead-L3` means "a notehead on the third line of the staff" and `notehead-S2` means "a notehead in the second space." The representation does not encode the *pitch* — whether `L3` corresponds to B (in treble clef) or D (in bass clef) depends on the clef context, which must be resolved by a separate post-processing step.

**Advantages:** Agnostic tokens map one-to-one to visible graphical elements, which can simplify the visual recognition task.

**Disadvantages:** A downstream semantic interpretation stage is required to produce musically meaningful output. Errors in that stage (especially clef or key-signature misinterpretation) can corrupt the entire transcription.

### Semantic Representation

A semantic representation encodes the *musical meaning* directly. The `**kern` format is inherently semantic: the token `4e-` unambiguously means "a quarter-note E-flat" regardless of which staff line the notehead occupied in the image. The model must implicitly learn the rules of music theory (e.g., that a notehead on the third line of a treble-clef staff is B, but D on a bass-clef staff), but modern Transformer architectures have proven more than equal to this task.

**Advantages:** The output is immediately musically meaningful — no post-processing is needed. It can be rendered back to notation or converted to MIDI directly.

**Disadvantages:** The model must learn to combine visual cues with notational conventions, which requires more training data and model capacity.

### Why This Work Uses Semantic Representation

The experiments in this thesis adopt the semantic `**kern` family for three reasons:

1. The end-to-end encoder-decoder architecture subsumes the visual-to-semantic mapping into a single learned function, eliminating the fragile two-stage pipeline that agnostic representations require.
2. The semantic output is directly usable for downstream tasks — it can be rendered back to notation by Verovio or converted to MIDI without an intermediate interpretation step.
3. The `bekern` tokenisation strategy (described in the next section) achieves a compact vocabulary while retaining full semantic content.

## Tokenisation Strategies: kern, ekern, and bekern

### The Problem with Standard **kern for Neural Networks

Standard `**kern` was designed for human readability and musicological analysis, not for neural network consumption. When each unique `**kern` symbol (e.g., `8e-JL`, `16f#/`, `[2.cc#`) is treated as a single indivisible token, the vocabulary can exceed 20,000 entries. Such a large vocabulary causes several problems for sequence models:

- **Data sparsity.** Many tokens appear only a handful of times in the training corpus, making their learned embeddings unreliable.
- **Gradient instability.** The softmax output layer must discriminate among tens of thousands of classes, which can slow convergence.
- **Poor generalisation.** The model cannot leverage the compositional structure of note symbols — it does not "know" that `8e-J` and `8e-L` share the same pitch and duration and differ only in beaming.

The core insight is that a musical note token is not atomic — it is composed of independent attributes (duration, pitch, accidental, beaming, articulation) whose combinations create the vocabulary explosion. Decomposing tokens into these components is analogous to moving from word-level to subword-level tokenisation in NLP: the sequences become longer, but the vocabulary shrinks dramatically and every token is well-represented in the training data.

### The Annotated Dataset Format

The datasets used in this work (published by the PRAIG research group at the University of Alicante, e.g., `grandstaff-ekern`) store their transcriptions in a pre-annotated format where two delimiter characters have been inserted into standard `**kern` tokens to mark decomposition boundaries:

- **`@`** separates components within the note body: duration, dotting, pitch, and accidental.
- **`·`** separates the note body from graphical modifiers: beaming markers (`L`, `J`), articulation, ties, and other notation symbols.

For example, the standard `**kern` token `8e-JL` (eighth note, E-flat, beam-end, beam-start) is stored in the dataset as `8@e@-·J·L`. This annotated form is not a separate format or standard — it is simply the raw data with explicit sub-token boundaries marked, enabling multiple tokenisation strategies from the same source.

### The Three Tokenisation Modes

The tokenisation strategy is selected at runtime through a configuration parameter (`krn_format`). All three modes consume the same annotated dataset; they differ only in how they handle the `@` and `·` delimiters:

**kern mode** — Removes both `@` and `·`, fusing everything back into monolithic tokens:

```python
krn = krn.replace("·", "").replace('@', '')
```

`8@e@-·J·L` → `8e-JL` (1 token)

Every unique combination of duration + pitch + accidental + beaming + articulation is a separate vocabulary entry. This produces the shortest sequences but the largest vocabulary (thousands of unique tokens).

**ekern mode** — Removes `@` (keeping the note body fused) but replaces `·` with spaces (splitting off graphical modifiers):

```python
krn = krn.replace("·", " ").replace('@', '')
```

`8@e@-·J·L` → `8e-` `J` `L` (3 tokens)

The note body (duration + pitch + accidental) stays as one token, but beaming, articulation, and other visual modifiers become independent tokens. This eliminates the combinatorial explosion from graphical variants while keeping note tokens semantically rich.

**bekern mode** — Replaces both `@` and `·` with spaces, decomposing everything into atomic components:

```python
krn = krn.replace("·", " ").replace("@", " ")
```

`8@e@-·J·L` → `8` `e` `-` `J` `L` (5 tokens)

Every duration, pitch letter, accidental, and graphical modifier is its own token. This produces the longest sequences but the smallest vocabulary — typically a few hundred entries, all of which are highly frequent in the training data.

### Comparison

| Property | kern | ekern | bekern |
|---|---|---|---|
| `@` delimiter | removed (fuse) | removed (fuse) | → space (split) |
| `·` delimiter | removed (fuse) | → space (split) | → space (split) |
| Typical vocabulary size | ~20,000+ | ~2,000–5,000 | ~200–500 |
| Sequence length | Shortest | Medium | Longest |
| Token–glyph correlation | Low | Medium | High |

The relationship between vocabulary size and sequence length is analogous to the word-level vs. character-level tradeoff in NLP. Kern mode is like word-level tokenisation: few tokens per sentence, but a massive dictionary where many words are rare. Bekern mode is like character-level tokenisation: many tokens per sentence, but a tiny alphabet where every symbol is frequent and well-learned.

The experiments in this thesis use **bekern** (`krn_format: "bekern"` in all configuration files), as it provides the best balance between vocabulary compactness and recognition accuracy. This choice follows the findings of Ríos-Vila et al., who demonstrated that bekern tokenisation yields the lowest Symbol Error Rate among the three variants.

## Implementation: From Score to Token Sequence

This section describes how a raw `**kern` score is transformed into the integer sequence that the model consumes during training and produces during inference.

### Linearisation: 2D Grid → 1D Token Stream

A `**kern` file is inherently two-dimensional (voices × time), but a sequence-to-sequence model outputs a one-dimensional token stream. The linearisation replaces whitespace characters with explicit structural tokens:

| Original character | Replacement token | Semantic meaning |
|---|---|---|
| Space (` `) | `<s>` | Separates simultaneous notes within one voice (chords) |
| Tab (`\t`) | `<t>` | Separates voices/staves (spine boundary) |
| Newline (`\n`) | `<b>` | Separates successive time-slices (temporal boundary) |

Three additional control tokens frame the sequence:

| Token | Index | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding for batch alignment; ignored by the loss function |
| `<bos>` | (assigned) | Beginning of sequence; always the first token |
| `<eos>` | (assigned) | End of sequence; signals the decoder to stop |

### Pre-processing Pipeline

Before tokenisation, the raw kern string undergoes several normalisation steps:

1. **Forbidden-token removal** (`clean_kern`): Lines containing structural tokens irrelevant to the musical content are deleted. The forbidden list includes: `*staff1`, `*staff2`, `*ped`, `*Xped`, `*tremolo`, `*Xtremolo`, `*tuplet`, `*Xtuplet`, `*cue`, `*Xcue`, `*rscale:1/2`, `*rscale:1`, `*kcancel`, `*below`. Lines consisting solely of `*` null interpretations are also removed.

2. **Bar-number stripping**: Numeric bar identifiers after barline tokens are removed (e.g., `=15` → `=`). Bar numbers are editorial additions with no musical content; stripping them reduces vocabulary size without losing information.

3. **Whitespace replacement**: Spaces, tabs, and newlines are replaced with `<s>`, `<t>`, and `<b>` respectively.

4. **Stem-direction removal**: The characters `/` and `\` (stem up / stem down) and their `·`-prefixed variants are removed, as stem direction is a typographic choice that carries no semantic musical information.

5. **Format-specific delimiter handling**: The `·` and `@` characters are processed according to the selected tokenisation mode (kern / ekern / bekern), as described in the previous section.

6. **Header removal** (full-page data only): The first four tokens of the parsed sequence (typically the `**ekern <t> **ekern <b>` spine-type declarations) are stripped because they are invariant across all scores in the dataset.

### Worked Example

Consider the following short two-voice kern score as stored in the annotated dataset:

```
**ekern         **ekern
*clefG2         *clefF4
*k[f#]          *k[f#]
*M3/4           *M3/4
=1              =1
4@c             4@E
4@d             4@F@#
=2              =2
*-              *-
```

**Step 1 — Clean:** No forbidden tokens are present; the score passes through unchanged.

**Step 2 — Strip bar numbers:** `=1` → `=`, `=2` → `=`.

**Step 3 — Replace whitespace and apply bekern splitting** (replace both `@` and `·` with spaces):

```
<bos> *clefG2 <t> *clefF4 <b> *k[f#] <t> *k[f#] <b> *M3/4 <t> *M3/4 <b> = <t> = <b> 4 c <t> 4 E <b> 4 d <t> 4 F # <b> = <t> = <b> *- <t> *- <eos>
```

Note how `4@c` became two tokens (`4` `c`) and `4@F@#` became three tokens (`4` `F` `#`) thanks to the `@` → space splitting.

**Step 4 — Vocabulary mapping:** Each unique string token is assigned an integer index. The index 0 is reserved for `<pad>`. A sample mapping:

| Token | Index |
|---|---|
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<t>` | 3 |
| `<b>` | 4 |
| `*clefG2` | 5 |
| `*clefF4` | 6 |
| `*k[f#]` | 7 |
| `*M3/4` | 8 |
| `=` | 9 |
| `4` | 10 |
| `c` | 11 |
| `E` | 12 |
| `d` | 13 |
| `F` | 14 |
| `#` | 15 |
| `*-` | 16 |

The final integer sequence:

```
[1, 5, 3, 6, 4, 7, 3, 7, 4, 8, 3, 8, 4, 9, 3, 9, 4, 10, 11, 3, 10, 12, 4, 10, 13, 3, 10, 14, 15, 4, 9, 3, 9, 4, 16, 3, 16, 2]
```

During training, this sequence is split into a *decoder input* (all tokens except `<eos>`) and a *label* (all tokens except `<bos>`), implementing the teacher-forcing paradigm.

### Vocabulary Statistics

The vocabulary files in the project reflect the compactness achieved by bekern tokenisation:

| Dataset | Tokenisation | Approximate vocabulary size |
|---|---|---|
| GrandStaff (single-system) | standard kern | ~8,000+ |
| FP GrandStaff (full-page) | bekern | ~200–300 |
| Polish Scores (full-page) | bekern | ~200–400 |

The order-of-magnitude reduction — achieved by decomposing composite symbols into atomic sub-tokens — is one of the key enablers of training stability and generalisation performance across all architectures evaluated in this thesis.

### Rendering and Round-Tripping

An important practical advantage of the kern-based formats is their compatibility with the Verovio rendering engine. The inverse of the linearisation process — replacing `<t>` with tabs, `<b>` with newlines, `<s>` with spaces, stripping `<bos>` and `<eos>`, and re-joining sub-tokens — produces a valid kern file that Verovio can render back into a musical score image. This round-tripping capability is exploited to generate unlimited synthetic training data: existing kern transcriptions from the dataset are sampled, optionally concatenated into multi-system pages, rendered to images via Verovio, and paired with their bekern token sequences as ground truth.
