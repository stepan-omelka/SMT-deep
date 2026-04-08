# Music Notation Formats

Before a neural network can learn to read sheet music, its visual content must be expressed as a target text sequence. The choice of representation dictates vocabulary size, sequence length, and model complexity. Standard digital music formats are unsuitable for end-to-end OMR: MIDI (MIDI Manufacturers Association, 1983) discards notational semantics entirely, while MusicXML (Good, 2001) and MEI (Roland, 2002) are excessively verbose for autoregressive generation. 

This thesis instead relies on the Humdrum `**kern` encoding (Huron, 1997) and its tokenisation variants, which offer compact, unambiguous serialisation of semantic musical data.

## The Humdrum **kern Encoding

Developed for computational musicology, `**kern` uses a flat text structure ideal for sequence modeling. It provides:

1. **Compactness:** A single token like `8e-JL` encodes duration (eighth), pitch (E), accidental (flat), and beaming (start/end) in just five characters.
2. **Tabular Structure:** Polyphony is encoded as a grid. Columns (spines) are tab-separated voices; rows are simultaneous time-slices.
3. **Semantic Representation:** It encodes direct musical meaning (e.g., E-flat) rather than agnostic geometric primitives (e.g., "notehead on line 3"), sidestepping the need for fragile post-processing steps.

### Polyphonic Representation Example
A simple C-major scale moving against an E-minor scale is represented as:

```
**kern      **kern
*clefG2     *clefF4
*M3/4       *M3/4
=1          =1
4c          4E
4d          4F#
4e          4G
*-          *-
```
The left spine is the upper staff and the right is the lower staff. Simultaneous notes across different voices align on the same text line. Chords within a single voice are simply separated by spaces (e.g., `4c 4e 4g`).

## Tokenisation Strategies: kern, ekern, and bekern

Standard `**kern` treats complex clusters (like `8e-JL`) as indivisible tokens. For neural networks, this causes a vocabulary explosion (>20,000 unique entries), leading to data sparsity and gradient instability. 

To solve this, the PRAIG research group datasets pre-annotate standard `**kern` data by inserting delimiters to mark sub-token boundaries (Ríos-Vila et al., 2023):
- **`@`** separates components within the semantic note body: duration `@` dotting `@` pitch `@` accidental.
- **`·`** separates the note body from graphical modifiers: beaming (`L`, `J`), ties (`[`, `]`), and articulations (`k`).

For example, `8e-JL` is stored in the dataset as `8@e@-·J·L`. This enables three different runtime tokenisation modes derived from the same source data:

1. **kern mode**: Removes both `@` and `·`, recombining into monolithic tokens (`8e-JL` = 1 token). Results in a very large vocabulary but short sequences.
2. **ekern mode**: Removes `@` but replaces `·` with a space. The note body stays fused, while visual modifiers are split off (`8e-` `J` `L` = 3 tokens). Provides a medium balance.
3. **bekern mode**: Replaces both `@` and `·` with spaces. Decomposes the symbol entirely into atomic sub-tokens (`8` `e` `-` `J` `L` = 5 tokens). Results in the smallest vocabulary but longest sequences.

### Comparison

| Property | kern | ekern | bekern |
|---|---|---|---|
| `@` delimiter | removed (fuse) | removed (fuse) | → space (split) |
| `·` delimiter | removed (fuse) | → space (split) | → space (split) |
| Typical vocabulary size | ~20,000+ | ~2,000–5,000 | ~200–500 |
| Sequence length | Shortest | Medium | Longest |

This thesis strictly uses the **bekern** format. Decomposing symbols into atomic primitives acts similarly to character-level tokenisation in NLP, shrinking the vocabulary to ~200–500 highly frequent symbols and dramatically stabilizing model training (Ríos-Vila et al., 2023).

## Implementation: 2D Grid → 1D Token Stream

A Sequence-to-Sequence neural network requires a flat 1D stream. The 2D `**kern` tabular grid is linearised into a 1D sequence through several pre-processing steps:

1. **Normalisation:** Irrelevant metadata (e.g., `*ped`), redundant bar numbers (`=15` → `=`), and purely typographic stem-direction markers (`/`, `\`) are stripped to minimize noise.
2. **Boundary Markers:** Formatting whitespace is mapped to explicit control tokens:
    - Space (` `) → `<s>` (separates chord notes in one voice)
    - Tab (`\t`) → `<t>` (separates different voices/spines)
    - Newline (`\n`) → `<b>` (separates successive time-slices)
3. **Framing:** The sequence is wrapped with `<bos>` (beginning) and `<eos>` (end).

**Linearisation Example (bekern mode):**
*Input (Annotated):*
```
4@c    4@E
4@d    4@F@#
```
*1D Output Sequence:*
`<bos> 4 c <t> 4 E <b> 4 d <t> 4 F # <eos>`

During inference, this process is fully reversible: the model's 1D prediction is mapped backward into standard `**kern`. Because it is standard `**kern`, the predicted transcription can be rendered back directly into sheet music via the Verovio engine (Pugin et al., 2014) without any intermediate semantic-lifting steps.

## References

* Good, M. (2001). *MusicXML: An internet-friendly method for sheet music exchange*. Interactive multimedia electronic journal of computer-enhanced learning, 3(2), 1-14.
* Huron, D. (1997). *Humdrum and Kern: Core repertoire for music research*. Computing in Musicology, 10, 11-36.
* MIDI Manufacturers Association. (1983). *MIDI 1.0 detailed specification*.
* Pugin, L., Zitellini, R., & Roland, P. (2014). *Verovio: A library for engraving MEI music notation into SVG*. Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR 2014), 107-112.
* Ríos-Vila, A., Calvo-Zaragoza, J., & Iñesta, J. M. (2023). *End-to-End Optical Music Recognition for Pianoform Music*. International Journal on Document Analysis and Recognition (IJDAR).
* Roland, P. (2002). *The Music Encoding Initiative (MEI)*. Proceedings of the First International Conference on Musical Applications Using XML, 55-59.
