import numpy as np
import operator

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a.flatten(), b.flatten().T)/(np.linalg.norm(a) * np.linalg.norm(b))


def top_n_sim(emb, embs, topn=5):
    """
    calculate cosine between query_emb and all_embs
    returns - list of tuples, each tuple of type (distance, index)
    :param query_sent : sentence 
    :param all_sents : all sentences in corpus   
    """
    # sort list according to cosine_similarity and return
    result = []
    for index, sent in enumerate(embs):
        
        sim = cosine_sim(emb, embs[index])
        result.append((sim, index))
        #break
    result.sort(key=operator.itemgetter(0), reverse=True)
    return result[:topn]

if __name__ == '__main__':
	
	for i, sent in enumerate(useful_sents[:5]):
		results = top_n_sim(embs[i], embs)
		print('sentence for comparasion - ', i,'.->',sent , '\n\n***************\n\n')
		for result in results:
			print(useful_sents[result[1]], '-->',result[0], '\n\n')
