# typing-classification
Classifying typing errors. For details on the features, read [here](features.md).
## To run (note that only English and Spanish are currently supported)
1. install required packages:  
	pandas, numpy, py2casefold
1. align the typed and original data:
       `python alignment.py input_file output_file [language]`
2. extract the features from the typing errors (detailed description of features is [here](features.md)):
       `python feature_extraction_parkinsons.py language`
3. classify the features
