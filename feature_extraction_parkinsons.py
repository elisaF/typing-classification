__author__ = 'elisa'
import logging
import pickle
import os
from feature_extraction_common_spanish import FeatureExtractionCommon

logger = logging.getLogger('feature_extraction_parkinsons')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('feature_extraction_parkinsons.log')
logger.addHandler(fh)


class FeatureExtractor:
    def __init__(self, language="english",
                 input_file='input'+os.path.sep+'errors_parkinsons_spanish.txt',
                 output_file='output'+os.path.sep+'output_spanish_parkinsons.csv',
                 pickle_file='output'+os.path.sep+'spanish_parkinsons.p',
                 lm_file="Typing-UT/LanguageModel/Spanish/SpanishEuroparl-noShift.lm"):
        self.input_file = input_file
        self.output_file = output_file
        self.pickle_file = pickle_file
        self.lm_file = lm_file
        self.fe = FeatureExtractionCommon(lm_file, language)

    def run_extractor(self):
        df = self.fe.create_data_frame(self.input_file, drop_column=False)
        df = self.extract_features(df)
        self.fe.save_data_frame(df, self.output_file)

    def extract_features(self, df):
        # preprocess data first

        # strip off trailing spaces
        df['Raw Typed'] = df['Raw Typed'].str.rstrip(' ')
        df['Intended'] = df['Intended'].str.rstrip(' ')

        df['Typed'] = df.apply(self.fe.create_typed_word, axis=1)
        df['Intended'] = df.apply(self.fe.clean_word, axis=1)
        temp_df = df.apply(self.fe.clean_context, axis=1)
        df['Error Context'], df['Position of word'] = zip(*temp_df)
        df = self.fe.drop_bad_rows(df)

        # now extract features
        df["diff_length"] = df["Intended"].str.len() - df["Typed"].str.len()
        df['error_start_typed'] = df.apply(self.fe.get_error_index, axis=1)
        df['error_start_intended'] = df['error_start_typed']

        df['error_end_typed'] = df.apply(self.fe.error_end_typed, axis=1)
        df['error_end_intended'] = df.apply(self.fe.error_end_intended, axis=1)
        df['edit_distance'] = df.apply(self.fe.get_edit_distance, axis=1)
        pickle.dump(df, open(self.pickle_file, "wb"))
        print('created  base features')

        df['keyboard_distance_typed_after'] = df.apply(self.fe.keyboard_distance_typed_after, axis=1)
        df['keyboard_distance_typed_before'] = df.apply(self.fe.keyboard_distance_typed_before, axis=1)
        df['keyboard_distance_same'] = df.apply(self.fe.keyboard_distance_same, axis=1)
        df['keyboard_distance_intended_after'] = df.apply(self.fe.keyboard_distance_intended_after, axis=1)
        df['keyboard_distance_intended_after2'] = df.apply(self.fe.keyboard_distance_intended_after2, axis=1)
        df['keyboard_distance_intended_before'] = df.apply(self.fe.keyboard_distance_intended_before, axis=1)
        df['keyboard_distance_intended_before2'] = df.apply(self.fe.keyboard_distance_intended_before2, axis=1)
        pickle.dump(df, open(self.pickle_file))
        print('First dump')

        df['ngram1_prob_typed'] = df.apply(self.fe.ngram1_prob_typed, axis=1)
        df['ngram2_prob_typed_before'] = df.apply(self.fe.ngram2_prob_typed_before, axis=1)
        df['ngram3_prob_typed_before'] = df.apply(self.fe.ngram3_prob_typed_before, axis=1)
        df['ngram4_prob_typed_before'] = df.apply(self.fe.ngram4_prob_typed_before, axis=1)
        df['ngram5_prob_typed_before'] = df.apply(self.fe.ngram5_prob_typed_before, axis=1)
        df['ngram2_prob_typed_after'] = df.apply(self.fe.ngram2_prob_typed_after, axis=1)
        df['ngram3_prob_typed_after'] = df.apply(self.fe.ngram3_prob_typed_after, axis=1)
        df['ngram4_prob_typed_after'] = df.apply(self.fe.ngram4_prob_typed_after, axis=1)
        df['ngram5_prob_typed_after'] = df.apply(self.fe.ngram5_prob_typed_after, axis=1)

        df['ngram1_prob_intended'] = df.apply(self.fe.ngram1_prob_intended, axis=1)
        df['ngram2_prob_intended_before'] = df.apply(self.fe.ngram2_prob_intended_before, axis=1)
        df['ngram3_prob_intended_before'] = df.apply(self.fe.ngram3_prob_intended_before, axis=1)
        df['ngram4_prob_intended_before'] = df.apply(self.fe.ngram4_prob_intended_before, axis=1)
        df['ngram5_prob_intended_before'] = df.apply(self.fe.ngram5_prob_intended_before, axis=1)
        df['ngram2_prob_intended_after'] = df.apply(self.fe.ngram2_prob_intended_after, axis=1)
        df['ngram3_prob_intended_after'] = df.apply(self.fe.ngram3_prob_intended_after, axis=1)
        df['ngram4_prob_intended_after'] = df.apply(self.fe.ngram4_prob_intended_after, axis=1)
        df['ngram5_prob_intended_after'] = df.apply(self.fe.ngram5_prob_intended_after, axis=1)
        pickle.dump(df, open(self.pickle_file, "wb"))
        print('ngrams done')
        return df

    def add_feature(self):
        df = pickle.load(open(self.pickle_file, "rb"))
        # add feature here
        pickle.dump(df, open(self.pickle_file, "wb"))
        self.fe.save_data_frame(df, self.output_file)
