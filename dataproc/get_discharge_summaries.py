"""
    Reads NOTEEVENTS file, finds the discharge summaries, preprocesses them and writes out the filtered dataset.
"""
import csv

from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm
from gensim.parsing.preprocessing import remove_stopwords

from constants import MIMIC_3_DIR
import re
#retain only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')

def write_discharge_summaries(out_file):
    notes_file = '%s/NOTEEVENTS.csv' % (MIMIC_3_DIR)
    print("processing notes file")
    with open(notes_file, 'r') as csvfile:
        with open(out_file, 'w') as outfile:
            print("writing to %s" % (out_file))
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            #header
            next(notereader)
            i = 0
            for line in tqdm(notereader):
                subj = int(line[1])
                category = line[6]
                if category == "Discharge summary":
                    note = line[10]
                    #tokenize, lowercase and remove numerics
                    #tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    #text = '"' + ' '.join(tokens) + '"'
                    text=remove_stopwords(udf_clean(note))
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')
                i += 1
    return out_file

def udf_clean(t):
    text=t.lower()
    # remove brackets
    text = re.sub('[\[].*?[\]]', '', text)

    # remove phrases
    phrases = ['admission date:', 'discharge date:', 'date of birth:', 'sex:', 'service:', 'patient', 'addendum']
    for phrase in phrases:
        text = re.sub(phrase, '', text)

    # remove punctuation
    text = re.sub('[^\w\s]', '', text)

    # split words
    text = text.split()

    # remove full string numbers
    text = [word for word in text if not word.isnumeric()]

    # remove single letter words
    text = [word for word in text if len(word) > 1]

    # convert to string
    text = ' '.join(text)

    return text
