import numpy as np
from sklearn import pipeline
from collections import Counter
import clean_data.preprocessing
from xgboost import XGBClassifier
from pandas import read_csv, set_option
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
#import dill as pickle # used to pickle lambda functions

from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================
# TRAIN THE BEST MODEL ON WHOLE DATASET (RETURN MODEL TO SAVE IT AS PICKLE FOR THE  WEB APP - PICKLE NEEDS TO BE
# GENERATED EXACTLY THIS WAY = USING THE pickle_model.py file)
# ======================================================================================================================

set_option('display.max_columns', None)


def dummy(token):  # needs to be outside of the train_model function
    '''
    Used for TF-IDF pre-processing and tokenizer
    :param token: token/word
    :return: the same token/word
    '''
    return token


def train_model():
    '''
    Train the best model on whole dataset
    :return: the best model trained on whole dataset
    '''
    # ======================================================================================================================
    # Read data
    # ======================================================================================================================

    tweets = read_csv('../dataset/hate_tweets.csv', encoding="utf8", index_col=False)


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
    # Classifier
    # ======================================================================================================================

    # Pipeline for data encoding and machine learning classifier 1
    min_df = 5  # term appears in less than 5 documents
    max_df = 0.8  # term appears in more than 80% of documents
    model = pipeline.Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=False, preprocessor=dummy, tokenizer=dummy, min_df=min_df, max_df=max_df,
                                  analyzer='word', ngram_range=(1, 3))),
        ('svd', TruncatedSVD(n_components=300)),
        # multi:softmax: multiclass classification using the softmax objective
        ('xgb', XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='multi:softmax',
                              nthread=4, random_state=27))

    ])


    # ======================================================================================================================
    # Set X and Y
    # ======================================================================================================================

    X = tweets['clean_text']
    Y = tweets['class']

    # replace labels numbers with their corresponding names
    print(Y)
    Y.replace({0: 'hate speech', 1: 'offensive language', 2: 'neither'}, inplace=True)
    print(Y)
    print(Y.value_counts())


    # ======================================================================================================================
    # AFTER FITTING THE MODEL, SAVE IT WITH PICKLE
    # ======================================================================================================================

    print(Y.value_counts())

    # count of the labels
    class_count = Counter(Y)

    # the max label count
    all_values = class_count.values()
    max_count = max(all_values)

    # undervalue class 2 ('neither', 3/4 of the majority class)
    class_weights = {'hate speech': max_count / class_count['hate speech'],
                     'offensive language': max_count / class_count['offensive language'],
                     'neither': 3/4 * (max_count / class_count['neither'])}

    print(class_weights)
    print(Y.shape[0])

    # assign the corresponding class weight for each individual data instance
    w_array = np.ones(Y.shape[0], dtype='float')
    for i, val in enumerate(Y):
        w_array[i] = class_weights[val]

    print(w_array)
    print(Y)

    # retrain model on whole dataset and save it
    model.fit(X, Y, xgb__sample_weight=w_array)

    return model
