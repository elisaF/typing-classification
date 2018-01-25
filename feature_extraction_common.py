from __future__ import division
__author__ = 'elisa'
import helper_functions as helper
import logging
import pandas as pd
from pyxdameraulevenshtein import damerau_levenshtein_distance
from keyboard_distance import KeyboardDistance

logger = logging.getLogger('feature_extraction_common')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('feature_extraction_common.log')
logger.addHandler(fh)


class FeatureExtractionCommon:
    def __init__(self, lm_file, language="english", low_freq_cutoff=25000, max_diff_length=15):
        self.language = language
        self.low_freq_cutoff = low_freq_cutoff
        self.max_diff_length = max_diff_length
        self.lm_file = lm_file
        self.keyboard_distance = KeyboardDistance(self.language)

    @staticmethod
    def save_data_frame(df, file_name):
        df.to_csv(file_name, encoding='utf-8', index=False)

    @staticmethod
    def create_data_frame(file_name, drop_column=True):
        df = pd.read_csv(file_name, sep='\t', encoding='utf-8', dtype={'Intended': str})
        df['Raw Typed'] = df['Raw Typed'].astype('unicode')
        df['Intended'] = df['Intended'].astype('unicode')

        # drop last column
        if drop_column:
            df.drop(df.columns[-1], axis=1, inplace=True)
        return df

    @staticmethod
    def create_typed_word(row):
        typed_word = ""
        word = row['Raw Typed']
        space = '}'
        backspace = '*'

        # word was not corrected
        if backspace not in word:
            if space not in word:
                typed_word = word
            else:
                typed_word = word[:word.index(space)]
        # word was corrected
        else:
            backspace_counter, num_backspace = 0, 0
            for char in word:
                if char == space:
                    break
                elif char == backspace:
                    # reset counter if needed
                    # i.e., we deleted the backspaced characters, and now encountered another backspace
                    # so we need to reset the counters
                    if num_backspace == backspace_counter:
                        backspace_counter, num_backspace = 0, 0
                    num_backspace += 1
                else:
                    if num_backspace == backspace_counter:
                        typed_word += char
                        # reset counters
                        backspace_counter, num_backspace = 0, 0
                    else:
                        backspace_counter += 1
        return typed_word

    @staticmethod
    def clean_word(row):
        intended = row['Intended'].replace('}', ' ')  # spaces at end of word should be ignore for Intended
        return intended

    def drop_bad_rows(self, df):
        nrows_orig = len(df)
        logger.debug('Original data frame length: ' + str(nrows_orig))

        # drop rows where intended is same as typed
        df = df.ix[~(df.Typed == df.Intended)]
        nrows_dup = len(df)
        logger.debug('Dropped no errors: ' + str(nrows_orig-nrows_dup))

        # drop rows where Typed is a blank
        df = df.ix[~(df.Typed == '')]
        nrows_blank = len(df)
        logger.debug('Dropped Typed blanks: ' + str(nrows_dup-nrows_blank))

        # drop rows where Typed starts with a backspace, since
        # this is an alignment error
        df = df.ix[~(df.Typed.str.startswith('*'))]
        nrows_misalgined = len(df)
        logger.debug('Dropped Typed misaligned: ' + str(nrows_blank-nrows_misalgined))

        # drops rows where Intended and Typed Context differ by more than cutoff, as
        # we assume this is either a bad subject or an alignment error
        df = df.ix[~(abs(df['Error Context'].str.len() - df['Intended Context'].str.len()) > self.max_diff_length)]
        nrows_toolong = len(df)
        logger.debug('Dropped sentence diff too big: ' + str(nrows_misalgined-nrows_toolong))
        return df

    @staticmethod
    def clean_context(row):
        orig_context = row['Raw Typed Context'].replace('}', ' ')
        orig_index = row['Original Position of word']
        backspace = '*'
        new_context = ""
        new_index = None
        new_index_counter = -1
        num_backspace, backspace_counter = 0, 0
        first_char_is_backspace = False

        for index, char in enumerate(orig_context):
            if char == backspace:
                # first misaligned character is a backspace
                if index == orig_index:
                    first_char_is_backspace = True
                # reset counter if needed
                # i.e., we deleted the replacement characters, and now encountered another backspace
                # so we need to reset the counters
                elif num_backspace == backspace_counter:
                    backspace_counter, num_backspace = 0, 0
                num_backspace += 1
            else:
                if num_backspace == backspace_counter or index == orig_index:  # don't delete the error word!
                    new_context += char
                    new_index_counter += 1
                    # reset counters
                    backspace_counter, num_backspace = 0, 0
                    if index == orig_index or first_char_is_backspace:
                        new_index = new_index_counter
                        first_char_is_backspace = False
                else:
                    backspace_counter += 1
        return new_context, new_index

    @staticmethod
    def get_error_index(row):
        error_start = None
        for index, letter in enumerate(row.Typed):
            # stop when we reach the end of an intended word
            # that is shorter than the typed word
            if index == len(row.Intended):
                error_start = index
                break
            # or a mismatched letter
            elif row.Intended[index] != letter:
                error_start = index
                break
            # or the end of a typed word that is the same as
            # intended but missing letters at the end
            elif index == len(row.Typed)-1:
                error_start = index+1
                break
        return error_start

    @staticmethod
    def error_end_intended(row):
        return max(len(row.Intended)-1, row.error_start_intended)  # account for error when last letter is omitted

    @staticmethod
    def error_end_typed(row):
        return max(len(row.Typed)-1, row.error_start_typed)  # account for error when last letter is omitted

    @staticmethod
    def get_edit_distance(row):
        return damerau_levenshtein_distance(row.Intended[row.error_start_intended:], row.Typed[row.error_start_typed:])

    @staticmethod
    def get_following_char_from_context(row, offset=0):
        # sometimes we don't have the full context, so just hard-code next letter to space
        if row['Position of word'] > len(row['Error Context']):
            logger.debug('Went past limit!')
            char = u' '
        # if this is the last word in the context, or we've gone past the end,
        # then hard-code next letter to period
        elif row['Position of word']+len(row['Typed'])+offset >= len(row['Error Context']):
            logger.debug('At the end of context!')
            if offset == 0:
                char = u'.'
            elif offset == 1:
                char = u' '
            else:
                logger.warning('Offset is greater than 1, so entering land of speculation. Will set following char to blank space.')
                char = u' '
        else:
            logger.debug('From context with length_typed: %s', str(len(row['Typed'])))
            logger.debug(row['Error Context'])
            char = row['Error Context'][row['Position of word']+len(row['Typed'])+offset]
        return char

    @staticmethod
    def get_previous_char_from_context(row, offset=1):
        # sometimes we don't have the full context, so just hard-code previous letter to space
        if row['Position of word'] > len(row['Error Context']):
            logger.debug('Went past limit!')
            char = u' '
        # if this is the first word in the wikipedia article, hard-code previous letter to space
        elif row['Position of word'] == 0:
            logger.debug('At the end of article!')
            char = u' '
        # get previous letter from context
        else:
            logger.debug('From context')
            logger.debug(row['Error Context'])
            logger.debug(row['Position of word'])

            char = row['Error Context'][row['Position of word']-offset]
        return char

    def get_mistyped_char_and_before(self, row, word, error_index, offset=1):
        logger.debug('ID: ' + str(row.ID))
        # error is after last letter of word (i.e., last letter was ommitted)
        if len(word) == error_index:
            first_char = self.get_following_char_from_context(row)
        else:
            first_char = word[error_index]

        # second char is before beginning of word
        if error_index - offset < 0:
            logger.debug('Error index: %s' % error_index)
            word_offset = offset - error_index
            logger.debug('Word offset: %s' % word_offset)
            second_char = self.get_previous_char_from_context(row, word_offset)
        else:
            second_char = word[error_index - offset]
        logger.debug('get_mistyped_char_and_before return: %s, %s' % (first_char, second_char))
        return first_char, second_char

    def get_mistyped_char_and_after(self, row, word, error_index, offset=1):
        logger.debug('get_mistyped_char_and_after enter: ID: ' + str(row.ID))
        # error is after last letter of word (i.e., last letter was ommitted)
        if len(word) == error_index:
            first_char = self.get_following_char_from_context(row)
            second_char = self.get_following_char_from_context(row, offset)
        else:
            first_char = word[error_index]

            # second char will go past end of word
            if len(word) <= error_index + offset:
                logger.debug('Error index: %s' % error_index)
                word_offset = error_index + offset - len(word)
                logger.debug('Word offset: %s' % word_offset)
                second_char = self.get_following_char_from_context(row, word_offset)
            else:
                second_char = word[error_index + offset]

        logger.debug('get_mistyped_char_and_after return: %s, %s' % (first_char, second_char))
        return first_char, second_char

    @staticmethod
    def get_mistyped_char(word, error_index):
        # error is after last letter of word (i.e., last letter was omitted)
        if len(word) == error_index:
            mistyped_char = u' '
        else:
            mistyped_char = word[error_index]
        logger.debug('get_mistyped_char return:  %s' % mistyped_char)
        return mistyped_char

    def keyboard_distance_typed_before(self, row):
        return self.keyboard_distance_before(row, row.Typed, row.error_start_typed)

    def keyboard_distance_intended_before(self, row):
        return self.keyboard_distance_before(row, row.Intended, row.error_start_intended)

    def keyboard_distance_intended_before2(self, row):
        return self.keyboard_distance_before(row, row.Intended, row.error_start_intended, 2)

    def keyboard_distance_before(self, row, word, error_index, offset=1):
        logger.debug(word)
        first_char, second_char = self.get_mistyped_char_and_before(row, word, error_index, offset)
        return self.keyboard_distance.calculate_distance(first_char, second_char)

    def keyboard_distance_typed_after(self, row):
        return self.keyboard_distance_after(row, row.Typed, row.error_start_typed)

    def keyboard_distance_intended_after(self, row):
        return self.keyboard_distance_after(row, row.Intended, row.error_start_intended)

    def keyboard_distance_intended_after2(self, row):
        return self.keyboard_distance_after(row, row.Intended, row.error_start_intended, 2)

    def keyboard_distance_after(self, row, word, error_index, offset=1):
        logger.debug(word)
        first_char, second_char = self.get_mistyped_char_and_after(row, word, error_index, offset)
        return self.keyboard_distance.calculate_distance(first_char, second_char)

    def keyboard_distance_same(self, row):
        logger.debug(row.Typed)
        logger.debug(row.Intended)
        # error is after last letter of word (i.e., last letter was ommitted)
        if len(row.Typed) == row.error_start_typed:
            first_char = self.get_following_char_from_context(row)
        else:
            first_char = row.Typed[row.error_start_typed]

        # error is after last letter of word (i.e., last letter was ommitted)
        if len(row.Intended) == row.error_start_intended:
            second_char = self.get_following_char_from_context(row)
        else:
            second_char = row.Intended[row.error_start_intended]
        logger.debug(first_char)
        logger.debug(second_char)
        return self.keyboard_distance.calculate_distance(first_char, second_char)

    def same_hand_after(self, row):
        first_char, second_char = self.get_mistyped_char_and_after(row, row.Intended, row.error_start_intended)
        return self.keyboard_distance.same_hand(first_char, second_char)

    def same_hand_before(self, row):
        first_char, second_char = self.get_mistyped_char_and_before(row, row.Intended, row.error_start_intended)
        return self.keyboard_distance.same_hand(first_char, second_char)

    def length_misaligned_sequence(self, row):
        logger.debug('length_misaligned_sequence enter: %s, %s' % (row.Typed, row.Intended))

        length_misaligned_typed = None
        length_misaligned_intended = None
        found_match = False
        offset = 0

        length_after_typed = len(row.Typed[row.error_start_typed:row.error_end_typed])
        length_after_intended = len(row.Intended[row.error_start_intended:row.error_end_intended])
        # substitution/migration/deletion of last letter
        if length_after_typed == length_after_intended:
            logger.debug('substitution or migration or deletion of last letter')
            while not found_match and row.error_start_typed+offset <= row.error_end_typed:
                offset += 1
                _, next_typed = self.get_mistyped_char_and_after(row, row.Typed, row.error_start_typed, offset)
                _, next_intended = self.get_mistyped_char_and_after(row, row.Intended, row.error_start_intended, offset)
                if next_typed == next_intended:
                    logger.debug('substitution')
                    found_match = True
                    length_misaligned_typed = offset
                    length_misaligned_intended = offset

            # start over and check for migration instead
            if not found_match:
                offset = 0
                while not found_match and row.error_start_typed+offset <= row.error_end_typed:
                    offset += 1
                    _, next_typed = self.get_mistyped_char_and_after(row, row.Typed, row.error_start_typed, offset)
                    mistyped_intended = self.get_mistyped_char(row.Intended, row.error_start_intended)
                    if next_typed == mistyped_intended:
                        logger.debug('migration')
                        found_match = True
                        length_misaligned_typed = offset
                        length_misaligned_intended = 0

            # start over and check for deletion of last letter instead
            if not found_match:
                offset = 0
                while not found_match and row.error_start_intended+offset <= row.error_end_intended:
                    offset += 1
                    mistyped_typed = self.get_mistyped_char(row.Typed, row.error_start_typed)
                    _, next_intended = self.get_mistyped_char_and_after(row, row.Intended, row.error_start_intended, offset)
                    if mistyped_typed == next_intended:
                        logger.debug('deletion of last letter')
                        found_match = True
                        length_misaligned_typed = 0
                        length_misaligned_intended = offset

        # insertion
        elif length_after_typed > length_after_intended:
            logger.debug('insertion')
            while not found_match and row.error_start_typed+offset <= row.error_end_typed:
                offset += 1
                _, next_typed = self.get_mistyped_char_and_after(row, row.Typed, row.error_start_typed, offset)
                mistyped_intended = self.get_mistyped_char(row.Intended, row.error_start_intended)
                if next_typed == mistyped_intended:
                    found_match = True
                    length_misaligned_typed = offset
                    length_misaligned_intended = 0

        # deletion
        elif length_after_typed < length_after_intended:
            logger.debug('deletion')
            while not found_match and row.error_start_intended+offset <= row.error_end_intended:
                offset += 1
                mistyped_typed = self.get_mistyped_char(row.Typed, row.error_start_typed)
                _, next_intended = self.get_mistyped_char_and_after(row, row.Intended, row.error_start_intended, offset)
                if mistyped_typed == next_intended:
                    found_match = True
                    length_misaligned_typed = 0
                    length_misaligned_intended = offset

        # combination of operations like substitution + deletion/insertion
        if not found_match:
            logger.debug('combination')
            length_misaligned_typed = 1+row.error_end_typed - row.error_start_typed
            length_misaligned_intended = 1+row.error_end_intended - row.error_start_intended

        logger.debug('length_misaligned_sequence exit: %s, %s' % (length_misaligned_typed, length_misaligned_intended))
        return length_misaligned_typed, length_misaligned_intended

    def get_ngram_before(self, row, word, error_index, offset):
        all_chars = self.get_mistyped_char(word, error_index)
        for index in range(1, offset+1):
            _, prev_char = self.get_mistyped_char_and_before(row, word, error_index, index)
            all_chars = prev_char + all_chars
        logger.debug('Word: %s' % word)
        logger.debug('get_ngram_before return: %s' % all_chars)
        return all_chars

    def get_ngram_after(self, row, word, error_index, offset):
        logger.debug('Word: %s' % word)
        all_chars = self.get_mistyped_char(word, error_index)
        for index in range(1, offset+1):
            _, after_char = self.get_mistyped_char_and_after(row, word, error_index, index)
            all_chars = all_chars + after_char
        logger.debug('Word: %s' % word)
        logger.debug('get_ngram_after return: %s' % all_chars)
        return all_chars

    def ngram1_prob_intended(self, row):
        logger.debug('ngram1_prob_intended_before, ID: %s' % row.ID)
        char = self.get_mistyped_char(row.Intended, row.error_start_intended)
        return helper.get_prob_chars(char, self.lm_file)

    def ngram2_prob_intended_before(self, row):
        logger.debug('ngram2_prob_intended_before, ID: %s' % row.ID)
        all_chars = self.get_ngram_before(row, row.Intended, row.error_start_intended, 1)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram3_prob_intended_before(self, row):
        logger.debug('ngram3_prob_intended_before, ID: %s' % row.ID)
        all_chars = self.get_ngram_before(row, row.Intended, row.error_start_intended, 2)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram4_prob_intended_before(self, row):
        logger.debug('ngram4_prob_intended_before, ID: %s' % row.ID)
        all_chars = self.get_ngram_before(row, row.Intended, row.error_start_intended, 3)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram5_prob_intended_before(self, row):
        logger.debug('ngram5_prob_intended_before, ID: %s' % row.ID)
        all_chars = self.get_ngram_before(row, row.Intended, row.error_start_intended, 4)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram2_prob_intended_after(self, row):
        logger.debug('ngram2_prob_intended_after, ID: %s' % row.ID)
        all_chars = self.get_ngram_after(row, row.Intended, row.error_start_intended, 1)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram3_prob_intended_after(self, row):
        logger.debug('ngram3_prob_intended_after, ID: %s' % row.ID)
        all_chars = self.get_ngram_after(row, row.Intended, row.error_start_intended, 2)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram4_prob_intended_after(self, row):
        logger.debug('ngram4_prob_intended_after, ID: %s' % row.ID)
        all_chars = self.get_ngram_after(row, row.Intended, row.error_start_intended, 3)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram5_prob_intended_after(self, row):
        logger.debug('ngram5_prob_intended_after, ID: %s' % row.ID)
        all_chars = self.get_ngram_after(row, row.Intended, row.error_start_intended, 4)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram1_prob_typed(self, row):
        logger.debug('ngram1_prob_typed, ID: %s' % row.ID)
        char = self.get_mistyped_char(row.Typed, row.error_start_typed)
        return helper.get_prob_chars(char, self.lm_file)

    def ngram2_prob_typed_before(self, row):
        logger.debug('ngram2_prob_typed_before, ID: %s' % row.ID)
        all_chars = self.get_ngram_before(row, row.Typed, row.error_start_typed, 1)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram3_prob_typed_before(self, row):
        logger.debug('ngram3_prob_typed_before, ID: %s' % row.ID)
        all_chars = self.get_ngram_before(row, row.Typed, row.error_start_typed, 2)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram4_prob_typed_before(self, row):
        logger.debug('ngram4_prob_typed_before, ID: %s' % row.ID)
        all_chars = self.get_ngram_before(row, row.Typed, row.error_start_typed, 3)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram5_prob_typed_before(self, row):
        logger.debug('ngram5_prob_typed_before, ID: %s' % row.ID)
        all_chars = self.get_ngram_before(row, row.Typed, row.error_start_typed, 4)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram2_prob_typed_after(self, row):
        logger.debug('ngram2_prob_typed_after, ID: %s' % row.ID)
        all_chars = self.get_ngram_after(row, row.Typed, row.error_start_typed, 1)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram3_prob_typed_after(self, row):
        logger.debug('ngram3_prob_typed_after, ID: %s' % row.ID)
        all_chars = self.get_ngram_after(row, row.Typed, row.error_start_typed, 2)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram4_prob_typed_after(self, row):
        logger.debug('ngram4_prob_typed_after, ID: %s' % row.ID)
        all_chars = self.get_ngram_after(row, row.Typed, row.error_start_typed, 3)
        return helper.get_prob_chars(all_chars, self.lm_file)

    def ngram5_prob_typed_after(self, row):
        logger.debug('ngram5_prob_typed_after, ID: %s' % row.ID)
        all_chars = self.get_ngram_after(row, row.Typed, row.error_start_typed, 4)
        return helper.get_prob_chars(all_chars, self.lm_file)
