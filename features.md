# Features
In order to describe the features, I first define 

## Keyboard distance: 
These features are based on the distance between two keys on the keyboard. First, the keys are mapped onto a grid. Depending on which language you use, a different keyboard grid is used.
To see the grids, check out keyboard_distance.py. The distance between two keys is then calculated as the following:
  -if keys are orthogonal to each other: 1
  -if keys are on secondary diagonal (top-right to bottom-left): 1
  -other location: sqrt(x^2 + y^2), where x is horizontal distance and y is vertical distance
  -if keys are not on map, then return the distance to the average location (middle of the keyboard)

NOTE: Characters with diacritics that are not on the keyboard are mapped to their diacriticless counterparts (e.g. é -> e)

Characters that are not on the map:
  -ø, æ, ß
  -Non-Latin characters, including Chinese, Greek, Cyrillic characters 

### Feature names:
-`keyboard_distance_typed` distance between mistyped key and the following typed key
-'keyboard_distance_typed_before` 
-`keyboard_distance_same` 
-`keyboard_distance_intended`
-`keyboard_distance_intended2`
-`keyboard_distance_intended_before`
-`keyboard_distance_intended_before2`
        
## Ngram probabilities:

-`ngram1_prob_typed`
-`ngram2_prob_typed_before`
-`ngram3_prob_typed_before`
-`ngram4_prob_typed_before`
-`ngram5_prob_typed_before`
-`ngram2_prob_typed`
-`ngram3_prob_typed`
-`ngram4_prob_typed`
-`ngram5_prob_typed`
-`ngram1`
-`ngram2_before`
-`ngram3_before`
-`ngram4_before`
-`ngram5_before`
-`ngram2`
-`ngram3`
-`ngram4`
-`ngram5`