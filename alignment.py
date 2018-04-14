import sys, re, pandas
import codecs

class ErrorAligner:
    """
      Use Needleman Wunsch algorithm to find optimal alignment for whole sentence.
      Typed character aligned to same character in intended - treated as correctly typed
      Typed character aligned to a different character in intended - treated as an error
      Typed character aligned to nothing in the intended - take intended to be the character in the intended string that follows the last correctly aligned intended character.
      Should Intended character aligned to nothing in the typed be ignored? - assumed to be an omission and thus ignored (justification - we are interested in errors of commission not errors of omission).
    """

    def __init__(self, input_file, output_file, language="english"):
        """
        Constructor.
        :param
            output_file: file name to write aligned errors to.
        """
        self.idmap = None
        if language.lower() == "english":
            self.idmap = self.create_idmap()
        self.input_file = input_file
        self.output_file = output_file
        self.gap_penalty = -2  # both for insertion and deletion

    def create_idmap(self):
        idmap = {'143': '1a', '144': '2a', '145': '2b', '146': '1b', '148': '3', '149': '4a', '150': '5a', '151': '5b',
                 '152': '4b', '153': '6a', '154': '6b', '155': '7a', '156': '7b', '158': '8a', '159': '8b', '160': '9a',
                 '161': '9b', '162': '10a', '163': '10b', '164': '11a', '165': '11b', '166': '12a', '167': '12b',
                 '168': '13a', '169': '13b', '170': '14a', '171': '14b', '172': '15a', '173': '15b', '174': '16a',
                 '175': '16b', '176': '17a', '177': '17b', '178': '18a', '179': '18b', '180': '19a', '181': '19b',
                 '182': '20a', '183': '20b', '184': '21a', '185': '22a', '186': '22b', '187': '21b', '188': '23',
                 '189': '24', '190': '25a', '191': '25b', '192': '26a', '193': '26b', '194': '27a', '195': '27b',
                 '196': '28a', '197': '28b', '198': '29a', '199': '29b', '200': '30a', '201': '30b', '202': '31a',
                 '203': '32a', '204': '32b', '205': '31b', '206': '33a', '207': '33b', '208': '34a', '209': '34b',
                 '210': '35a', '211': '35b', '212': '36a', '213': '36b', '214': '37a', '215': '37b', '216': '38a',
                 '217': '39a', '218': '39b', '219': '38b', '220': '40a', '221': '40b', '222': '41a', '223': '41b',
                 '224': '42a', '225': '42b', '226': '43a', '227': '43b', '229': '44a', '230': '44b'}
        return idmap

    def outputforanalysis(self, align1, align2, seq2, itempartid, times, typedseq):
        # String is reversed once words identified. As beginning of words should be point
        # of alignment and thus the natural place to segment, or at least I assumed.
        # This affects output however. Consider other way around?
        # align1 = align1[::-1]    #reverse sequence 1
        # align2 = align2[::-1]    #reverse sequence 2

        i = 0
        targetword = ""
        typedword = ""
        typedindex = 0
        wordcount = 0
        while i < len(align1):
            position = str(len(typedseq) - typedindex)
            if align1[i] != "^" and align2[i] == "}":
                if typedword != targetword:
                    typedwordtoprint = typedword[::-1]
                    targetwordtoprint = targetword[::-1]
                    if typedword != "":
                        typedstringposition = len(typedseq) - typedindex
                        charindex = int(
                            typedstringposition + int(finderrorlocation(targetwordtoprint, typedwordtoprint)))
                        text = itempartid + "-" + str(wordcount) + "\t" + typedwordtoprint + "\t" + targetwordtoprint + \
                               "\t" + position + "\t" + typedseq + "\t" + seq2 + "\t" + str(times[charindex])+"\n"
                        with codecs.open(self.output_file, 'ab', 'utf-8') as out:
                            out.write(text.replace('.0', ''))

                targetword = ""
                typedword = ""
                typedindex += 1
                wordcount += 1
                
            elif i == len(align1) - 1:
                if align1[i] != "^":
                    typedword += align1[i]
                if align2[i] != "^":
                    targetword += align2[i]
                if typedword != targetword:
                    typedwordtoprint = typedword[::-1]
                    targetwordtoprint = targetword[::-1]
                    if typedword != "":
                        if align1[i] == "^":
                            typedstringposition = len(typedseq) - typedindex 
                        else:
                            typedstringposition = len(typedseq) - typedindex - 1
                        
                        errorloc = int(finderrorlocation(targetwordtoprint, typedwordtoprint))
                        charindex = int(typedstringposition + errorloc)

                        if charindex == 0:                        
                            text = itempartid + "-" + str(wordcount) + "\t" + typedwordtoprint + "\t" + \
                                   targetwordtoprint + "\t" + position + "\t" + typedseq + "\t" + seq2 + "\tNA\n"
                       
                        else:                      
                            text = itempartid + "-" + str(wordcount) + "\t" + typedwordtoprint + "\t" + \
                                   targetwordtoprint + "\t" + position + "\t" + typedseq + "\t" + seq2 + "\t" + \
                                   str(times[charindex]) + "\n"
                       
                        with codecs.open(self.output_file, 'ab', 'utf-8') as out:                            
                            out.write(text.replace('.0', ''))
                          
            else:
                if align1[i] != "^":
                    typedword += align1[i]
                    typedindex += 1
                if align2[i] != "^":
                    targetword += align2[i]
            i += 1

    # The function needle is adapted from the following :
    # https://github.com/alevchuk/pairwise-alignment-in-python
    def needle(self, seq1, seq2, itempartid, times):
        m, n = len(seq1), len(seq2)  # length of two sequences

        # Generate DP table and traceback path pointer matrix
        score = zeros((m + 1, n + 1))  # the DP table

        # Calculate DP table
        for i in range(0, m + 1):
            score[i][0] = self.gap_penalty * i
        for j in range(0, n + 1):
            score[0][j] = self.gap_penalty * j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score[i - 1][j - 1] + self.match_score(seq1[i - 1], seq2[j - 1])
                delete = score[i - 1][j] + self.gap_penalty
                insert = score[i][j - 1] + self.gap_penalty
                score[i][j] = max(match, delete, insert)

        # Traceback and compute the alignment
        align1, align2 = '', ''
        i, j = m, n  # start from the bottom right cell
        while i > 0 and j > 0:  # end touching the top or the left edge
            score_current = score[i][j]
            score_diagonal = score[i - 1][j - 1]
            score_up = score[i][j - 1]
            score_left = score[i - 1][j]

            if score_current == score_diagonal + self.match_score(seq1[i - 1], seq2[j - 1]):
                align1 += seq1[i - 1]
                align2 += seq2[j - 1]
                i -= 1
                j -= 1
            elif score_current == score_left + self.gap_penalty:
                align1 += seq1[i - 1]
                align2 += '^'
                i -= 1
            elif score_current == score_up + self.gap_penalty:
                align1 += '^'
                align2 += seq2[j - 1]
                j -= 1

        # Finish tracing up to the top left cell
        while i > 0:
            align1 += seq1[i - 1]
            align2 += '^'
            i -= 1
        while j > 0:
            align1 += '^'
            align2 += seq2[j - 1]
            j -= 1

        self.outputforanalysis(align1, align2, seq2, itempartid, times, seq1)

    def parse_errors(self):
        # write header
        header = "ID\tRaw Typed\tIntended\tOriginal Position of word\tRaw Typed Context\tIntended Context\tIKI_FOR_ERROR\n"
        with codecs.open(self.output_file, 'wb', 'utf-8') as out:
            out.write(header)
            
        lastpartid = -1
        lastsentid = -1
        lastrespid = -1
        lastsenttext = -1
        lasttime = -1
        typedtext = ""

        keypresstimes = {}
        charcount = 0

        f = pandas.ExcelFile(self.input_file)
        df = f.parse(f.sheet_names[0])
        l = 0

        while l < len(df):
            time = float(df.ix[l][0])
            if charcount == 0:
                iki = 0
            else:
                iki = time - lasttime

            typedchar = unicode(df.ix[l][1]).rstrip('"').lower()
            typedchar = re.sub(r'backspace', '*', typedchar)
            typedchar = re.sub(r'","', ',', typedchar)
            typedchar = re.sub(r' ', '}', typedchar)
            respid = df.ix[l][2]
            if self.idmap:
                partid = self.idmap[str(df.ix[l][4])]
            else:
                partid = df.ix[l][4]
            sentid = df.ix[l][5]
            senttext = unicode(df.ix[l][6]).rstrip().rstrip('"').lower()
            senttext = re.sub(r'^"|"$', '', senttext)
            senttext = re.sub(r' ', '}', senttext)
            lastid = '-'.join([unicode(lastpartid), unicode(lastsentid)])
            keypresstimes[charcount] = iki

            if (sentid != lastsentid or partid != lastpartid or respid != lastrespid) and l > 1:
                # Change IF statement below to check sentence not already processed by making sure
                # lastid not in a done vector, as could break if different participants version of
                # the same sentence were to follow each other
                if senttext != lastsenttext:
                    self.needle(typedtext, lastsenttext, lastid, keypresstimes)
                    keypresstimes = {}
                    charcount = 0
                typedtext = ""
                typedtext += typedchar
            else:
                typedtext += typedchar

            lastpartid = partid
            lastsentid = sentid
            lastrespid = respid
            lastsenttext = senttext
            lasttime = time
            l += 1
            charcount += 1

    def match_score(self, alpha, beta):
        match_award = 1
        mismatch_penalty = -1

        if alpha == beta:
            return match_award
        elif alpha == '^' or beta == '^':
            return self.gap_penalty
        else:
            return mismatch_penalty


# zeros() was originally from NumPy.
# This version is implemented by alevchuk 2011-04-10
def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval


def finderrorlocation(intendedword, typedword):
    e = 0
    while e < len(typedword):
         
        if e >= len(intendedword):
       
            return int(e)
        elif typedword[e] != intendedword[e]:
       
            return int(e)
       
        e = e + 1
    return int(len(typedword))


if __name__ == '__main__':
    if len(sys.argv) > 4 or len(sys.argv) < 3:
        print("FORMAT: input_file output_file [language]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if len(sys.argv) > 3:
        language = sys.argv[3]
        aligner = ErrorAligner(input_file, output_file, language)
    else:
        aligner = ErrorAligner(input_file, output_file)
    aligner.parse_errors()
