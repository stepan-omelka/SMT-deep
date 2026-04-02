# Music Notation Formats

Before a neural network can learn to read sheet music, the visual content of a score must be expressed as a target text sequence — the "language" the model is trained to produce. The choice of this target representation is far from trivial. It determines the size of the model's vocabulary, the length of its output sequences, and the degree to which the textual tokens correlate with identifiable graphical elements in the image. A representation that is too coarse collapses musical detail; one that is too fine explodes the vocabulary into tens of thousands of rarely-seen symbols that are difficult to learn. This chapter surveys the landscape of music notation formats, beginning with the widely-known industry standards, proceeding to the foundational Humdrum `**kern` encoding, and culminating in the specialised tokenisation variants — `kern`, `ekern`, and `bekern` — that the Sheet Music Transformer employs.

## Standard Digital Music Formats

Three dominant families of digital music representation exist today, each designed for a different purpose. Understanding why none of them is used directly as the target format of the SMT clarifies the design decisions that led to the `bekern` encoding.

### MIDI (Musical Instrument Digital Interface)

MIDI, introduced in 1983, is a communication protocol rather than a file format in the traditional sense. A MIDI file does not store audio; it stores *instructions* — which note to play, at what velocity, for how long, on which channel. Conceptually, a MIDI message is closer to a mechanical piano roll than to a printed score: it encodes the *performance* of a piece without preserving its *notation*. A pianist pressing middle C with moderate force and releasing it after 500 milliseconds produces a `Note On (channel=0, note=60, velocity=80)` followed, 500 ms later, by a `Note Off (channel=0, note=60)`.

This performance-centric design entails several consequences that make MIDI unsuitable as a ground-truth format for OMR:

- **No notational semantics.** MIDI cannot distinguish between a dotted quarter note and a quarter note tied to an eighth note — both produce the same duration. Likewise, enharmonic spellings (C♯ vs. D♭) are collapsed into a single pitch number.
- **No visual layout information.** Concepts such as staff assignment, stem direction, beaming groups, and clef context do not exist in MIDI.
- **Low resolution.** Classic MIDI 1.0 uses 7-bit values (0–127) for pitch and velocity, which is insufficient to represent the full richness of Western notation, let alone microtonal or non-Western traditions.

While MIDI remains dominant in music production and real-time performance, it discards precisely the information that OMR seeks to recover: the visual and structural meaning of the printed score.

### MusicXML

MusicXML is an XML-based interchange format designed to transfer musical scores between notation software such as MuseScore, Finale, Sibelius, and Dorico. It explicitly encodes notational elements — pitch (with correct enharmonic spelling), duration, voice assignment, clefs, key signatures, time signatures, articulations, dynamics, and layout instructions. A MusicXML document may be organised part-wise (measures nested inside parts) or time-wise (parts nested inside measures), and the format supports both uncompressed `.musicxml` and compressed `.mxl` variants.

MusicXML is comprehensive and widely supported (over 270 applications), yet its XML verbosity makes it a poor fit for neural sequence generation:

- **Extreme sequence length.** Even a short phrase generates hundreds of XML tags. An autoregressive decoder that must predict `<note>`, `<pitch>`, `<step>`, `C`, `</step>`, `<octave>`, `4`, `</octave>`, `</pitch>`, `<duration>`, `1`, `</duration>`, `<type>`, `quarter`, `</type>`, `</note>` for a single quarter-note C4 wastes the model's capacity on syntactic scaffolding rather than musical content.
- **Ambiguous serialisation.** The same musical passage can be validly encoded in multiple ways (part-wise vs. time-wise; different orderings of attributes), which complicates training because the model must learn to produce one canonical ordering.

MusicXML excels as a software interchange format, but it was not designed for the compact, unambiguous serialisation that sequence-to-sequence models require.

### MEI (Music Encoding Initiative)

MEI is an open-source, community-driven XML schema developed by and for the musicological research community. While superficially similar to MusicXML, MEI places a much stronger emphasis on scholarly metadata: editorial interventions, variant readings across manuscript sources, provenance, and critical apparatus. It supports notation systems far beyond Common Western Music Notation, including mensural notation, neumes, and tablature. The rendering library Verovio — which the SMT project uses to generate synthetic training images — natively consumes MEI input.

MEI shares MusicXML's verbosity problem and extends it: its richer metadata capabilities make documents even longer. For OMR training, where the goal is to produce a compact sequence of musical tokens, MEI is unnecessarily complex.

### Comparative Summary

| Property | MIDI | MusicXML | MEI |
|---|---|---|---|
| Primary purpose | Performance / playback | Software interchange | Scholarly archiving |
| Encodes pitch spelling | No | Yes | Yes |
| Encodes visual layout | No | Partially | Yes |
| Encodes editorial metadata | No | No | Yes |
| Typical verbosity | Low | High | Very high |
| Suitability for seq2seq OMR | Poor | Poor | Poor |

None of these standard formats provides the combination of properties needed for end-to-end OMR: compact token sequences, unambiguous serialisation, preservation of both pitch semantics and visual layout, and a manageable vocabulary size. The Humdrum `**kern` family of encodings was designed to fill exactly this niche.

## The Humdrum **kern Foundation

### Origins and Design Philosophy

Humdrum is a set of command-line tools and data representations for music analysis, developed by David Huron at Ohio State University in the 1990s. Its core data representation, `**kern`, was designed not for music production or notation interchange, but for *computational musicology* — systematic, quantitative analysis of musical corpora. This analytical orientation gives `**kern` two properties that happen to make it exceptionally well-suited for OMR:

1. **Compactness.** A single `**kern` token like `8e-J` encodes duration (`8` = eighth note), pitch (`e`), accidental (`-` = flat), and beaming (`J` = end of beam group) in just four characters. The same information requires over a dozen XML tags in MusicXML.
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

This tabular structure is the key advantage of `**kern` for polyphonic OMR: it preserves the temporal alignment of voices that is visible in the score image, while remaining a flat text format that can be serialised into a 1D token stream.

## Agnostic vs. Semantic Representations

In the broader OMR literature, output representations fall into two fundamentally different categories, and understanding this dichotomy is essential for appreciating why the SMT uses a semantic format.

### Agnostic Representation

An agnostic representation describes the *visual appearance* of the score without interpreting its musical meaning. Tokens describe graphical primitives and their spatial positions on the staff:

```
notehead-L3  stem-up  flag-8th  notehead-S2  accidental-sharp  ...
```

Here, `notehead-L3` means "a notehead on the third line of the staff" and `notehead-S2` means "a notehead in the second space." The representation is agnostic because it does not tell you the *pitch* — whether `L3` is a B (in treble clef) or a D (in bass clef) depends on the clef context, which must be resolved by a separate post-processing step.

**Advantages:** Agnostic tokens map one-to-one to visible graphical elements, which can simplify the visual recognition task — every token the model produces corresponds to something it can directly "see" in the image.

**Disadvantages:** The representation requires a downstream semantic interpretation stage to produce musically meaningful output. Errors in this second stage (especially clef or key-signature misinterpretation) can corrupt the entire transcription. Furthermore, certain musical concepts (ties across barlines, voice assignments in polyphonic textures) are difficult to express without semantic context.

### Semantic Representation

A semantic representation encodes the *musical meaning* directly. The `**kern` format is inherently semantic: the token `4e-` unambiguously means "a quarter-note E-flat" regardless of which staff line the notehead occupied in the image. The model must learn to combine visual cues (clef, key signature, staff position) with notational conventions to produce correct semantic tokens.

**Advantages:** The output is immediately musically meaningful — no post-processing is needed. The representation naturally handles transposing instruments, enharmonic equivalence, and cross-staff notation.

**Disadvantages:** The model must implicitly learn the rules of music theory (e.g., that a notehead on the third line of a treble-clef staff is B, but D on a bass-clef staff). This requires more training data and a more capable model, but modern Transformer architectures have proven equal to the task.

### Why the SMT Uses Semantic Representation

The Sheet Music Transformer adopts the semantic `**kern` family because:

1. The end-to-end architecture subsumes the visual-to-semantic mapping into a single learned function, eliminating the fragile two-stage pipeline that agnostic representations require.
2. The semantic format produces output that is directly usable for music analysis, playback, and further processing — it can be rendered back to notation by Verovio or converted to MIDI without an intermediate interpretation step.
3. The `bekern` tokenisation (described in the next section) achieves a compact vocabulary while retaining full semantic content.

## The kern → ekern → bekern Tokenisation Hierarchy

Standard `**kern` was designed for human readability and musicological analysis, not for neural network consumption. When each unique `**kern` symbol (e.g., `8e-JL`, `16f#/`, `[2.cc#`) is treated as a single indivisible token, the vocabulary can exceed 20,000 entries. Such a large vocabulary causes several problems for sequence models:

- **Data sparsity.** Many tokens appear only a handful of times in the training corpus, making their embeddings poorly trained.
- **Gradient instability.** The softmax output layer must discriminate among tens of thousands of classes, which can slow convergence and cause vanishing gradients.
- **Poor generalisation.** The model cannot leverage the compositional structure of symbols — it does not "know" that `8e-J` and `8e-L` share the same pitch and duration and differ only in beaming.

The PRAIG research group at the University of Alicante addressed these problems by introducing two progressively finer tokenisation schemes: **ekern** (extended kern) and **bekern** (basic extended kern). Together with standard `kern`, they form a hierarchy of increasing granularity.

### kern (Standard Tokenisation)

In the standard tokenisation, composite symbols are kept intact but auxiliary graphical markers — the stem-direction indicator (`/` or `\`) and the sub-token separator (`·`) — are simply stripped away. The token `8e-·J/` becomes `8e-J`: a single vocabulary entry encoding duration, pitch, accidental, and beam-end in one unit.

In the codebase (`utils.py`, line 103):
```python
krn = krn.replace("·", "").replace('@', '')
```

**Vocabulary size:** Large (thousands of unique tokens), because every distinct combination of duration + pitch + accidental + beaming + articulation is a separate entry.

### ekern (Extended Kern)

ekern splits composite symbols at graphical boundaries. The `·` delimiter is replaced by a space (creating separate tokens), while the `@` marker (which separates certain editorial annotations) is discarded. The token `8e-·J` becomes two tokens: `8e-` and `J`. Note tokens remain unified — duration and pitch stay fused — but beaming, articulation, and other graphical modifiers become independent tokens.

In the codebase (`utils.py`, line 105):
```python
krn = krn.replace("·", " ").replace('@', '')
```

**Vocabulary size:** Moderate. The number of distinct note-body tokens (duration + pitch + accidental) is still substantial, but the combinatorial explosion from beaming and articulation variants is eliminated.

### bekern (Basic Extended Kern)

bekern takes decomposition to its logical conclusion: *every* sub-component becomes a separate token. Both `·` and `@` delimiters are replaced by spaces. The token `8e-·J@slur` becomes four tokens: `8`, `e-`, `J`, `slur` (the exact decomposition depends on how the original dataset was annotated).

In the codebase (`utils.py`, lines 106–107):
```python
krn = krn.replace("·", " ").replace("@", " ")
```

**Vocabulary size:** Small (typically a few hundred unique tokens). Every duration, every pitch-letter, every accidental, and every graphical modifier is a known, frequently-occurring vocabulary entry. This dramatically improves embedding quality and model generalisation.

**The tradeoff:** bekern produces the longest output sequences (more tokens per measure), which increases the computational cost of autoregressive decoding. However, the reduced vocabulary and improved per-token training signal more than compensate, and bekern has been shown to yield the best OMR accuracy in the PRAIG group's experiments.

### Comparative Summary

| Property | kern | ekern | bekern |
|---|---|---|---|
| `·` delimiter | removed | → space (split) | → space (split) |
| `@` delimiter | removed | removed | → space (split) |
| Typical vocabulary size | ~20,000+ | ~2,000–5,000 | ~200–500 |
| Sequence length | Shortest | Medium | Longest |
| Token–glyph correlation | Low | Medium | High |

The SMT project is configured to use **bekern** by default (`krn_format: "bekern"` in all experiment configuration files), as it provides the best balance between vocabulary compactness and recognition accuracy.

## Implementation Format: From Score to Token Sequence

This section describes, in concrete detail, how a raw `**kern` score is transformed into the integer sequence that the model consumes during training and produces during inference. Every operation described here is implemented in the project's `utils.py` and `data.py` files.

### Linearisation: 2D Grid → 1D Token Stream

A `**kern` file is inherently two-dimensional (voices × time), but a sequence-to-sequence model outputs a one-dimensional token stream. The linearisation works by replacing whitespace characters with explicit structural tokens:

| Original character | Replacement token | Semantic meaning |
|---|---|---|
| Space (` `) | `<s>` | Separates sub-tokens within a voice's time-slice (e.g., chords or multiple-stops) |
| Tab (`\t`) | `<t>` | Separates concurrently-sounding voices (spine boundary) |
| Newline (`\n`) | `<b>` | Separates successive time-slices (temporal boundary) |

Three additional control tokens frame the sequence:

| Token | Index | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding for batch alignment; ignored by the loss function |
| `<bos>` | (assigned) | Beginning of sequence; always the first token |
| `<eos>` | (assigned) | End of sequence; signals the decoder to stop |

### Pre-processing Pipeline

Before tokenisation, the raw kern string undergoes several normalisation steps, implemented in the `clean_kern` and `parse_kern` functions:

1. **Forbidden-token removal** (`clean_kern`): Lines containing structural tokens that are irrelevant to the musical content are deleted entirely. The forbidden list includes: `*staff1`, `*staff2`, `*ped`, `*Xped`, `*tremolo`, `*Xtremolo`, `*tuplet`, `*Xtuplet`, `*cue`, `*Xcue`, `*rscale:1/2`, `*rscale:1`, `*kcancel`, `*below`. Lines consisting solely of `*` null interpretations are also removed.

2. **Bar-number stripping**: The regex `(?<=\=)\d+` removes numeric bar identifiers after barline tokens (e.g., `=15` becomes `=`). Bar numbers are editorial additions with no musical content; stripping them reduces vocabulary size without losing information.

3. **Whitespace replacement**: Spaces, tabs, and newlines are replaced with `<s>`, `<t>`, and `<b>` respectively.

4. **Stem-direction removal**: The characters `/` and `\` (stem up / stem down) and their `·`-prefixed variants (`·/`, `·\`) are removed, as stem direction is a typographic choice that carries no semantic musical information.

5. **Format-specific delimiter handling**: The `·` and `@` characters are processed according to the selected format (kern / ekern / bekern), as described in the previous section.

6. **Header removal** (full-page data only): In `prepare_fp_data`, the first four tokens of the parsed sequence (typically `**kern`, `<t>`, `**kern`, `<b>` — the spine-type declarations) are stripped because they are invariant across all scores in the dataset and carry no discriminative information.

### Worked Example

Consider the following short two-voice kern score:

```
**kern    **kern
*clefG2   *clefF4
*k[f#]    *k[f#]
*M3/4     *M3/4
=1        =1
4c        4E
4d        4F#
=2        =2
*-        *-
```

**Step 1 — Clean:** No forbidden tokens are present, so the score passes through unchanged.

**Step 2 — Strip bar numbers:** `=1` becomes `=`, `=2` becomes `=`.

**Step 3 — Replace whitespace and apply bekern splitting:**

```
<bos> *clefG2 <t> *clefF4 <b> *k[f#] <t> *k[f#] <b> *M3/4 <t> *M3/4 <b> = <t> = <b> 4 c <t> 4 E <b> 4 d <t> 4 F# <b> = <t> = <b> *- <t> *- <eos>
```

(Note: in this simplified example without `·` or `@` markers, bekern behaves identically to kern. In real datasets that contain annotated sub-token structure, the `·` splitting produces finer decomposition.)

**Step 4 — Vocabulary mapping:** Each unique string token is assigned an integer index by the `make_vocabulary` function. The index 0 is reserved for `<pad>`. A sample mapping might be:

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
| `F#` | 14 |
| `*-` | 15 |

The final integer sequence fed to the model is:

```
[1, 5, 3, 6, 4, 7, 3, 7, 4, 8, 3, 8, 4, 9, 3, 9, 4, 10, 11, 3, 10, 12, 4, 10, 13, 3, 10, 14, 4, 9, 3, 9, 4, 15, 3, 15, 2]
```

During training, this sequence is split into a *decoder input* (all tokens except `<eos>`) and a *label* (all tokens except `<bos>`), implementing the teacher-forcing paradigm.

### Vocabulary Statistics

The vocabulary files saved in the project's `vocab/` directory reflect the compactness achieved by bekern tokenisation:

| Dataset | Vocabulary name | Approximate vocabulary size |
|---|---|---|
| GrandStaff (single-system, standard kern) | `GrandStaff` | ~8,000+ |
| FP GrandStaff (full-page, bekern) | `FP_GrandStaff_BeKern` | ~200–300 |
| Polish Scores (full-page, bekern) | `Polish_Scores_BeKern` | ~200–400 |

The order-of-magnitude reduction from standard kern to bekern — achieved by decomposing composite symbols into atomic sub-tokens — is one of the key enablers of the SMT's training stability and generalisation performance.

### Rendering and Round-Tripping

An important practical advantage of the bekern format is its compatibility with the Verovio rendering engine. The inverse of the linearisation process — replacing `<t>` with tabs, `<b>` with newlines, `<s>` with spaces, stripping `<bos>` and `<eos>`, and re-joining sub-tokens — produces a valid kern file that Verovio can render back into a musical score image. This round-tripping capability is exploited by the `SynthGenerator.py` module to generate unlimited synthetic training data: existing kern transcriptions from the dataset are sampled, optionally concatenated into multi-system pages, rendered to images via Verovio, and paired with their bekern token sequences as ground truth.
