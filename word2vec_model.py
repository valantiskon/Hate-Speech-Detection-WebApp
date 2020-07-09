import os
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full


from gensim.models.doc2vec import TaggedDocument, Doc2Vec



from gensim.models import Word2Vec
from sklearn.preprocessing import scale
from pandas import np
from sklearn.feature_extraction.text import TfidfVectorizer



class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.id2word = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.id2word = Dictionary.load(self.path)

    def save(self):
        self.id2word.save(self.path)

    def fit(self, documents, labels=None):
        self.id2word = Dictionary(documents)
        self.save()
        return self

    def transform(self, documents):
        for document in documents:
            docvec = self.id2word.doc2bow(document)
            # convert the sparse representation into a NumPy array in order to use in sklearn ML models
            yield sparse2full(docvec, len(self.id2word))





# ======================================================================================================================




from sklearn.pipeline import make_union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin

class DocLength(TransformerMixin):
    def fit(self, X, y=None):  # some boilerplate
        return self

    def transform(self, X):
        return [
            # note that we needed both positive and negative
            # feature - otherwise for linear model there won't
            # be a feature to show in a half of the cases
            [len(doc) % 2, not len(doc) % 2]
            for doc in X
        ]

    def get_feature_names(self):
        return ['is_odd', 'is_even']

vec = make_union(DocLength(), CountVectorizer(ngram_range=(1,2)))
te4 = TextExplainer(vec=vec).fit(doc[:-1], predict_proba_len)

print(te4.metrics_)
te4.explain_prediction(target_names=twenty_train.target_names)






# ======================================================================================================================
# word2vec
# ======================================================================================================================

# Used for computing the mean of word2vec and implementing the transform function
def buildWordVector(word2vec_model, tweet, size, tf_idf):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet:
        try:
            vec += word2vec_model[word].reshape((1, size)) * tf_idf[word]
            count += 1.
        except KeyError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


def labelizeTweets(tweet, label_type):
    LabeledSentence = gensim.models.doc2vec.TaggedDocument

    labelized = []
    for i, v in enumerate(tweet):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# word2vec Vector Size
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

vector_size = 100

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Train set
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
train_encoded_tweets = labelizeTweets(x_train, 'TRAIN')

# sg: CBOW if 0, skip-gram if 1
# ‘min_count’ is for neglecting infrequent words.
# negative (int) – If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
# window: number of words accounted for each context( if the window size is 3, 3 word in the left neighborhood and 3 word in the right neighborhood are considered)
model = Word2Vec(size=vector_size, min_count=5, sg=1)
model.build_vocab([x.words for x in train_encoded_tweets])
model.train([x.words for x in train_encoded_tweets], total_examples=len(train_encoded_tweets), epochs=10)

vectorizer1 = TfidfVectorizer(lowercase=False, analyzer=lambda x: x, min_df=7)
vectorizer1.fit_transform([x.words for x in train_encoded_tweets])

tfidf = dict(zip(vectorizer1.get_feature_names(), vectorizer1.idf_))
train_vecs_w2v = np.concatenate([buildWordVector(model, tweet, vector_size, tfidf) for tweet in
                                map(lambda x: x.words, train_encoded_tweets)])
x_train = scale(train_vecs_w2v)
print(x_train)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test set
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_encoded_tweets = labelizeTweets(x_test, 'TRAIN')
vectorizer1.transform([x.words for x in test_encoded_tweets])

tfidf = dict(zip(vectorizer1.get_feature_names(), vectorizer1.idf_))
train_vecs_w2v = np.concatenate([buildWordVector(model, tweet, vector_size, tfidf) for tweet in
                                map(lambda x: x.words, test_encoded_tweets)])
x_test = scale(train_vecs_w2v)
print(x_test)


'''
    def fit(self, documents, labels=None):
        corpus = [list(tokenize(doc)) for doc in documents]
        self.id2word = [
            TaggedDocument(words, ['d{}'.format(idx)])
            for idx, words in enumerate(corpus)
        ]
        self.save()
        return self

    def transform(self, documents):
        for document in documents:
            docvec = self.id2word.doc2bow(document)
            yield sparse2full(docvec, len(self.id2word))

model = Doc2Vec(corpus, size=5, min_count=0)
print(model.docvecs[0])
# [ 0.01797447 -0.01509272  0.0731937   0.06814702 -0.0846546 ]
'''