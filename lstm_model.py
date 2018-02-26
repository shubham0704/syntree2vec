
from __future__ import print_function
import numpy as np

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential

def sample(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def eval_on_lstm(tokenised_sents, word_model, max_sentence_len, test_ratio=0.2):
    
    sentences = [sentence for sentence in tokenised_sents if len(sentence) < max_sentence_len]
    pretrained_weights = word_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape
    def word2idx(word):
        try:
            idx = word_model.wv.vocab[word].index
        except:
            print("word: {} not in vocab using default word card\n".format(word))
            idx = 0
        return idx
    
    def idx2word(idx):
        return word_model.wv.index2word[idx]
    
    total = len(sentences)
    train_size = int(total * (1 - test_ratio))
    test_size = total - train_size
    
    train_x = np.zeros([train_size, max_sentence_len], dtype=np.int32)
    train_y = np.zeros([train_size], dtype=np.int32)
    test_x = np.zeros([test_size, max_sentence_len], dtype=np.int32)
    test_y = np.zeros([test_size], dtype=np.int32)

    for i, sentence in enumerate(sentences[:train_size]):
        for t, word in enumerate(sentence[:-1]):
            train_x[i, t] = word2idx(word)
        train_y[i] = word2idx(sentence[-1])
    for i, sentence in enumerate(sentences[train_size:]):
        for t, word in enumerate(sentence[:-1]):
            test_x[i, t] = word2idx(word)
        test_y[i] = word2idx(sentence[-1])
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
    model.add(LSTM(units=emdedding_size))
    model.add(Dense(units=vocab_size))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    def generate_next(text, num_generated=10):
        word_idxs = [word2idx(word) for word in text.lower().split()]
        for i in range(num_generated):
            prediction = model.predict(x=np.array(word_idxs))
            idx = sample(prediction[-1], temperature=0.7)
            word_idxs.append(idx)
        return ' '.join(idx2word(idx) for idx in word_idxs)
    
    def on_epoch_end(epoch, _):
        print('\nGenerating text after epoch: %d' % epoch)
        texts = [
        'chess',
        'is',
        'fantasy',
        'by',
        'steve',
        'lasting'
        ]
        for text in texts:
            sample = generate_next(text)
            print('%s... -> %s' % (text, sample))
            
    model.fit(train_x, train_y,
          batch_size=128,
          epochs=20,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
    
    scores = model.evaluate(test_x, test_y, verbose=0)
    print("Accuracy of the model is {}".format(scores))