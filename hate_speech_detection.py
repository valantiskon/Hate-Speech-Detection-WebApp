import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import clean_data.preprocessing
from sklearn import pipeline, model_selection, metrics, svm
from pandas import read_csv, set_option
import numpy as np
from sklearn.utils import class_weight
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from xgboost import XGBClassifier
#import dill as pickle # used to pickle lambda functions
from pytagcloud import create_tag_image, create_html_data, make_tags, LAYOUT_MIX
from pytagcloud.lang.counter import get_tag_counts
from pytagcloud.colors import COLOR_SCHEMES
import webbrowser

from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================

set_option('display.max_columns', None)


# ======================================================================================================================
# Dynamic wordcloud using HTML
# ======================================================================================================================


def wordcloud(self):
    words = nltk.word_tokenize(self._contents)
    doc = " ".join(d for d in words[:70])

    tags = make_tags(get_tag_counts(doc), maxsize=100)
    data = create_html_data(tags, (1600,1200), layout=LAYOUT_MIX, fontname='Philosopher', rectangular=True)
    webbrowser.open('sid.jpeg')


# ======================================================================================================================
# Read data
# ======================================================================================================================

tweets = read_csv('dataset/hate_tweets.csv', index_col=False)


# ======================================================================================================================
# Pre-processing
# ======================================================================================================================

# create object of class preprocessing to clean data
reading = clean_data.preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True, only_verbs_nouns=False)


# clean text using preprocessing.py (clean_Text function)
tweets['clean_text'] = tweets.tweet.progress_map(reading.clean_text)

# Drop the rows that contain empty captions
# inplace=True: modify the DataFrame in place (do not create a new object) - returns None
tweets.drop(tweets[tweets['clean_text'].progress_map(lambda d: len(d)) < 1].index, inplace=True)  # drop the rows that contain empty captions
# data[data['clean_text'].str.len() < 1]  # alternative way
tweets.reset_index(drop=True, inplace=True)  # reset index needed for dataframe access with indices


# ======================================================================================================================
# Set X and Y
# ======================================================================================================================

X = tweets['clean_text']
Y = tweets['class']

print(Y.value_counts())


# ======================================================================================================================
# Train/Test set split
# ======================================================================================================================

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=0)


# ======================================================================================================================
# Classifier
# ======================================================================================================================


def dummy(token):
    return token


# Pipeline for data encoding and machine learning classifier 1
min_df = 5  # term appears in less than 5 documents
max_df = 0.8  # term appears in more than 80% of documents
model = pipeline.Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False, preprocessor=dummy, tokenizer=dummy, min_df=min_df, max_df=max_df,
                              analyzer='word', ngram_range=(1, 3))),
    #('tfidf', TfidfVectorizer(lowercase=False, max_df=max_df, max_features=max_features, analyzer='word', tokenizer=lambda x: x, preprocessor=lambda x: x)),
    #('tfidf', TfidfVectorizer(preprocessor=' '.join, stop_words='english')), # use default tokenizer on sentence created by joining tokens using preprocessor

    ('svd', TruncatedSVD(n_components=1000)),

#    ('scaler', MinMaxScaler()),

    #('svm', svm.SVC(kernel='rbf', C=1000, gamma=0.1, decision_function_shape='ovr'))
    #('svm', svm.SVC(kernel='rbf', C=100, gamma=0.1, decision_function_shape='ovr'))
#    ('svm', svm.SVC(kernel='poly', C=100, gamma=0.1, class_weight='balanced', decision_function_shape='ovr', probability=True))

    #('svm', RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None))
    #('svm', LogisticRegression(solver="liblinear", C=300, max_iter=300))

    # multi:softmax: multiclass classification using the softmax objective
    ('xgb', XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='multi:softmax',
                          nthread=4, random_state=27))

]) # F1 Score: 0.6741140000713671

# {'svm__C': 100, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}           F1 Score: 0.6479
# {'svm__C': 100, 'svm__gamma': 0.1, 'svm__kernel': 'poly'}          F1 Score: 0.6646
# {'svm__C': 1000, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}          F1 Score: 0.6574


# ======================================================================================================================
# Tune weight for imbalanced classification in XGB Classifier
# ======================================================================================================================

class_weights = list(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))

print(class_weights)
print(y_train.shape[0])

# assign the corresponding class weight for each individual data instance
w_array = np.ones(y_train.shape[0], dtype='float')
for i, val in enumerate(y_train):
    w_array[i] = class_weights[val]

print(w_array)
print(y_train)

model.fit(x_train, y_train, xgb__sample_weight=w_array)

# ======================================================================================================================

# Train model
#model.fit(x_train, y_train)

# Make prediction on test set
y_predicted = model.predict(x_test)
print(set(y_predicted))

# Evaluate model with Recall, Precision and F1-Score metrics
acc = metrics.accuracy_score(y_test, y_predicted)
prec = metrics.precision_score(y_test, y_predicted, average='macro')
rec = metrics.recall_score(y_test, y_predicted, average='macro')
f1 = metrics.f1_score(y_test, y_predicted, average='macro')

print('\nResults:')
print('Accuracy: {}'.format(acc))

from sklearn.metrics import balanced_accuracy_score
print("balanced accuracy: %.2f" % balanced_accuracy_score(y_test, y_predicted))

print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))


# ======================================================================================================================
# AFTER FITTING THE MODEL, SAVE IT WITH PICKLE
# ======================================================================================================================

# retrain model on whole dataset and save it
model.fit(X, Y)

# Saving model to disk
pickle.dump(model, open('model.pkl', 'wb'))

'''
# Load the model to make predictions
model = pickle.load(open('model.pkl','rb'))
y_predicted = model.predict(x_test)
'''