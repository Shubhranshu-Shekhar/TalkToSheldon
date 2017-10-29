## Disclaimer: I have not given credit to sources from which we took the w2v code and modified. Will update it with right credits soon.

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser
from collections import defaultdict
import re
import numpy as np
import gensim, os
from sklearn.metrics.pairwise import cosine_similarity
from BBTDatasetReader import BBTDatasetReader

stops = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 'her',
 'hers',
 'herself',
 'it',
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 'should',
 'now']


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vecFname, bigramTranformerFname):
        try:
            self.w2v = Word2Vec.load(word2vecFname)
            self.word2weight = None
            #        self.dim = len(word2vec.itervalues().next())
            self.dim = self.w2v.vector_size
            self.max_idf = None
            self.bigram_transformer = Phraser.load(bigramTranformerFname)
        except Exception, e:
            print 'something wrong:' + str(e)

    @staticmethod
    def sentenceToWordlist(sentence, removeStopwords=True):
        sentence_text = sentence
        sentence_text = re.sub("[^a-zA-Z0-9]", " ", sentence_text)
        words = sentence_text.lower().split()
        if removeStopwords:
            words = [w for w in words if not w in stops]
        words = [w for w in words if len(w) > 1]
        return (words)

    def return_max_idf(self):
        return self.max_idf

    def returnTokens(self, x):
        return x

    def fit(self, X):
        X_prime1 = X
        #X_prime1 = [self.sentenceToWordlist(sent) for sent in X]
        # print X_prime1
        X_prime = self.bigram_transformer[X_prime1]
        # print X_prime
        tfidf = TfidfVectorizer(analyzer=self.returnTokens)
        tfidf.fit(X_prime)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        self.max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            self.return_max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform_org(self, X):
        return np.array([
                            np.mean([self.w2v[w] * self.word2weight[w]
                                     for w in words if w in self.w2v] or
                                    [np.zeros(self.dim)], axis=0)
                            for words in X
                            ])

    def transform(self, X):
        X_prime1 = [self.sentenceToWordlist(sent) for sent in X]
        # print X_prime
        X_prime = self.bigram_transformer[X_prime1]
        return np.array([
                            np.mean([self.w2v[w] * self.word2weight[w]
                                     for w in sent if w in self.w2v] or
                                    [np.zeros(self.dim)], axis=0)
                            for sent in X_prime
                            ])

    def most_similar(self, input_sentence, candidate_sentences, character_sentences):
        max_sim = -1000
        best_candidate_idx = -1

        zipped_sentences = zip(candidate_sentences, character_sentences)
        #input_sentence_tokens = TfidfEmbeddingVectorizer.sentenceToWordlist(input_sentence)

        for idx, candidate_sentence in enumerate(candidate_sentences):
            #candidate_sentence_tokens = TfidfEmbeddingVectorizer.sentenceToWordlist(candidate_sentence)
            if not candidate_sentence or candidate_sentence[0] == "" : continue
            candidate_sentence = " ".join(candidate_sentence)
            sim = cosine_similarity(self.transform([input_sentence]), self.transform([candidate_sentence]))
            # print candidate_sentence, sim
            if sim[0][0] > max_sim:
                max_sim = sim[0][0]
                # print candidate_sentence
                best_candidate_idx = idx
        print best_candidate_idx
        return zipped_sentences[best_candidate_idx][1], zipped_sentences[best_candidate_idx][0], max_sim


class MySentences(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.utterances = BBTDatasetReader(self.file_name)

    def prepare(self):
        return [TfidfEmbeddingVectorizer.sentenceToWordlist(utterance.others_utterance) for 
        utterance in self.utterances.read(character_name="Sheldon")]


class MyHeroEmbedding():
    def __init__(self, traindatapath, phraser_path, w2v_path):
        self.sentences = MySentences(traindatapath).prepare()
        #self.sentences = [s for s in sentences]
        self.phraser_path = phraser_path
        self.w2v_path = w2v_path


    def train(self):
        bigram_transformer = gensim.models.Phrases(self.sentences, min_count=2, threshold=2)
        bigram_transformer.save(self.phraser_path)
        bigram_model = gensim.models.Word2Vec(self.sentences, hs=1,
                            negative=0, alpha=0.001, window=30, sample=1e-3, sg=1, min_count=5, 
                            workers=4, iter=200, size=200)
        bigram_model.save(self.w2v_path)
        # unigram model best found params 
        #gensim.models.Word2Vec(bigram_transformer[self.sentences], hs=1,
        #                    negative=0, alpha=0.001, window=30, sample=1e-3, sg=0, min_count=5, 
        #                    workers=4, iter=200, size=200)


def main():
    traindatapath = '../data/corpus.json'
    phraser_path = '../data/hero_phraser' # Phraser for BBT script
    w2v_path = '../data/hero_word2vec' # Sheldon model

    myemb = MyHeroEmbedding(traindatapath, phraser_path, w2v_path)
    
    #myemb.train()
    tfidfw2v = TfidfEmbeddingVectorizer(myemb.w2v_path, myemb.phraser_path)
    tfidfw2v_model = tfidfw2v.fit(myemb.sentences)

    similarity = cosine_similarity(tfidfw2v_model.transform(["hello"]), tfidfw2v_model.transform(["hi"]))
    print similarity
    return tfidfw2v

if __name__ == '__main__':
    main()
