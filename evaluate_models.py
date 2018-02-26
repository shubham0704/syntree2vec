
from sentence_loader import lazy_load
from lstm_model import eval_on_lstm
import gensim

sizes = [1,2,4,8]
tokenized_sents = []
sents = []
for size in sizes:
    ts, s = lazy_load(chunk_size=(10240)*size)
    tokenized_sents.extend(ts)
    sents.extend(s)
    word_model = gensim.models.Word2Vec.load('./data/syncode/syncode_model_'+str(size)+'MB')
    eval_on_lstm(tokenized_sents,word_model, 40)
    print "\n done with syncode "+ str(size)+"\n"
    word_model = gensim.models.Word2Vec.load('./data/w2v/w2v_model_'+str(size)+'MB')
    eval_on_lstm(tokenized_sents,word_model, 40)
    print "\n done with word2vec "+ str(size)+"\n"