# typing-classification
Classifying typing errors from Parkinson's patients

## To run (note that only English and Spanish are currently supported)
1. align the typed and original data:
      `python alignment.py input_file output_file [language]`
2. extract the features from the spelling errors
      `python feature_extraction_parkinsons.py language`
3. classify the features

