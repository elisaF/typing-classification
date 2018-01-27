# Features
For describing these features, we define the mistyped character as the first character of the misspelled word that is out of alignment when comparing the misspelled and correct word. The aligned character is the corresponding character in the correct word. For example, in the pair _fpr_-_for_, the mistyped character is _p_ and the aligned character is _o_.

## Keyboard distance: 
The layout of the QWERTY keyboard is mapped to a grid-like matrix. The distance between the same key is 0, between orthogonal keys is 1, between keys that are on a secondary diagonal (top-right to bottom-left) is 1, and otherwise is calculated as the euclidean distance. These distances are meant to favor keys that share an edge with each other.
Before mapping, a character is normalized, unless the un-normalized character exists in the matrix (for example, é is mapped to e, ñ is mapped to ñ for Spanish). If a character is not found in the matrix, then the average location (middle of the keyboard) is used as its location, and the distance is calculated as above. 
Characters that are not on the map are: ø, æ, ß, non-Latin characters including Chinese, Greek, Cyrillic characters.

### Feature names:
- `keyboard_distance_same` distance between the mistyped character and the aligned character
- `keyboard_distance_typed_after` distance between the mistyped character and the typed character 1 position after
- `keyboard_distance_typed_before` distance between the mistyped character and the typed character 1 position before
- `keyboard_distance_intended_after` distance between the mistyped character and the aligned character 1 position after
- `keyboard_distance_intended_before` distance between the mistyped character and the aligned character 1 position before
- `keyboard_distance_intended_after2` distance between the mistyped character and the aligned character 2 positions after
- `keyboard_distance_intended_before2` distance between the mistyped character and the aligned character 2 positions before
        
## Ngram probabilities:
A character-level language model is created using SriLM 1.7.2, one from the English Europarl corpus (Koehn, 2005) combined with all the English patient data and the other from the Spanish version. Due to a bug in SriLM where the `-tolower` option mangles the multi-byte utf-8 encoded characters, the input is case-folded in a preprocessing step (see `create_vocab_for_lm` in `helper_functions.py`).

### Feature names:
- `ngram1_prob_typed` log probability of mistyped character
- `ngram1_prob_intended` log probability of aligned character
- `ngram2_prob_typed_before` log probability of character 2-gram, ending at mistyped character
- `ngram3_prob_typed_before` log probability of character 3-gram, ending at mistyped character
- `ngram4_prob_typed_before` log probability of character 4-gram, ending at mistyped character
- `ngram5_prob_typed_before` log probability of character 5-gram, ending at mistyped character
- `ngram2_prob_typed_after` log probability of character 2-gram, starting at mistyped character
- `ngram3_prob_typed_after` log probability of character 3-gram, starting at mistyped character
- `ngram4_prob_typed_after` log probability of character 4-gram, starting at mistyped character
- `ngram5_prob_typed_after` log probability of character 5-gram, starting at mistyped character
- `ngram2_intended_before` log probability of character 2-gram, ending at aligned character
- `ngram3_intended_before` log probability of character 3-gram, ending at aligned character
- `ngram4_intended_before` log probability of character 4-gram, ending at aligned character
- `ngram5_intended_before` log probability of character 5-gram, ending at aligned character
- `ngram2_intended_after` log probability of character 2-gram, starting at aligned character
- `ngram3_intended_after` log probability of character 3-gram, starting at aligned character
- `ngram4_intended_after` log probability of character 4-gram, starting at aligned character
- `ngram5_intended_after` log probability of character 5-gram, starting at aligned character 