"""
Author: Shubham Bhardwaj
Github: shubham0704
"""

import re
import sys
import nltk
import string
import logging
import numpy as np
from unidecode import unidecode
from memory_profiler import profile
from nltk.tokenize import WordPunctTokenizer

word_punct_tokenizer = WordPunctTokenizer()

stp_words = ["lrb", "rrb", "sjg","``", "''", ',']


f = open("./data/wiki_complete_dump_2008.txt.tokenized")
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")



@profile
def read_in_chunks(file_object, chunk_size=102400):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data



def lazy_load(file_object=f,chunk_size=102400):
	"""returns 2-D sents list with each sent in word tokenized format"""

	try:
		chunk = next(read_in_chunks(f, chunk_size))
	except Exception as e:
		logging.warning("File read fully: ", e)
		f.close()
		return None

	sents = tokenizer.tokenize(chunk.decode('utf-8'))

	tok_sents = []
	sentences = []
	table = string.maketrans("","")
	for sent in sents:
	    tok_sent = [word for word in word_punct_tokenizer.tokenize(unidecode(sent)) if word not in string.punctuation and word.isalnum() and len(word)>1 and word not in stp_words and not(re.match('^[\'-]', word))]
	    if tok_sent:
	        tok_sents.append(tok_sent)
	        sentence = " ".join(tok_sent)
	        sentence = sentence.translate(table, string.punctuation)
	        sentences.append(sentence)

	return tok_sents, sentences

def test():
    full_x = []
    full_y = []
    sizes = [1,2,4,8]
    for size in sizes:
        x,y = lazy_load(f, 1048576*size)
        full_x.extend(x)
        full_y.extend(y)
        del x
        del y
        #print sys.getsizeof(full_x)/1048576.0
        #print sys.getsizeof(full_x)/1048576.0
if __name__ == '__main__':
    test()
