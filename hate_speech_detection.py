import os
from string import Template
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
import clean_data.preprocessing
from sklearn import pipeline, model_selection, metrics, svm
from pandas import read_csv, set_option
import numpy as np
from sklearn.utils import class_weight
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
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


def interactive_wordcloud(all_texts):
    '''

    :param all_texts:
    :return:
    '''
    flat_text = []
    for text in all_texts:
        for word in text:
            flat_text.append(word)
    print(flat_text)

    counts = Counter(flat_text).items()
    print(counts)

    sorted_wordscount = sorted(counts, key=lambda tup: tup[1])[:200]  # sort and select the top 200 words counts
    print(sorted_wordscount)
    # Running get_tag_counts result in error UnicodeDecodeError: 'charmap' codec can't decode byte 0xaa in position 90: character maps to <undefined>
    # This is because in file stopwords.py, that is called by counter.py (contains code for get_tag_counts), the stopwords are not read in utf-8
    tags = make_tags(sorted_wordscount, maxsize=100)
    print('tags', tags)
    data = create_html_data(tags, size=(1600, 800), layout=LAYOUT_MIX, fontname='Philosopher', rectangular=True)
    print('data', data)

    # ======================================================================================================================
    # Write wordcloud on HTML file
    # ======================================================================================================================

    template_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out/template.html'), 'r')
    html_template = Template(template_file.read())

    context = {}

    tags_template = '<li class="cnt" style="top: %(top)dpx; left: %(left)dpx; height: %(height)dpx;"><a class="tag %(cls)s" href="#%(tag)s" style="top: %(top)dpx;\
            left: %(left)dpx; font-size: %(size)dpx; height: %(height)dpx; line-height:%(lh)dpx;">%(tag)s</a></li>'

    context['tags'] = ''.join([tags_template % link for link in data['links']])
    context['width'] = data['size'][0]
    context['height'] = data['size'][1]
    context['css'] = "".join("a.%(cname)s{color:%(normal)s;}\
            a.%(cname)s:hover{color:%(hover)s;}" %
                             {'cname': k,
                              'normal': v[0],
                              'hover': v[1]}
                             for k, v in data['css'].items())

    html_text = html_template.substitute(context)

    test_output = os.path.join(os.getcwd(), 'out')
    html_file = open(os.path.join(test_output, 'cloud.html'), 'w')
    html_file.write(html_text)
    html_file.close()
    '''
    # Write HTML String to file.html
    with open("wordcloud.html", "w") as file:
        file.write(data)
    '''
    #webbrowser.open('sid.jpeg')


# generate wordclouds
def wordcloud(all_texts, image_file_name):
    print('all_texts', all_texts)
    flat_text = []
    for text in all_texts:
        for word in text:
            flat_text.append(word)
    print(flat_text)

    counts = Counter(flat_text)
    print(counts)

    wordcloud = WordCloud(width=1600, height=800, max_words=200, background_color="black").generate_from_frequencies(counts)

    # Display the generated image
    plt.figure(figsize=(16, 8), facecolor='k')
    plt.tight_layout(pad=0)  # shrink the size of the border
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    plt.savefig(image_file_name + '.png', facecolor='k', bbox_inches='tight', dpi=300)



# ======================================================================================================================
# Read data
# ======================================================================================================================

tweets = read_csv('dataset/hate_tweets.csv', encoding="utf8", index_col=False)


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
# Wordclouds for each label
# ======================================================================================================================

hate_speech_text = tweets[tweets['class'] == 0]
wordcloud(hate_speech_text['clean_text'], 'hate_speech')

offens_lang_text = tweets[tweets['class'] == 1]
wordcloud(offens_lang_text['clean_text'], 'offens_lang')

neither_text = tweets[tweets['class'] == 2]
wordcloud(neither_text['clean_text'], 'neither')


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

    ('svd', TruncatedSVD(n_components=300)),

#    ('scaler', MinMaxScaler()),

#    ('svm', svm.SVC(kernel='poly', C=100, gamma=0.1, class_weight='balanced', decision_function_shape='ovr', probability=True))

    # multi:softmax: multiclass classification using the softmax objective
    ('xgb', XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='multi:softmax',
                          nthread=4, random_state=27))

])  # F1 Score: 0.6968215126026323

# {'svm__C': 100, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}           F1 Score: 0.6479
# {'svm__C': 100, 'svm__gamma': 0.1, 'svm__kernel': 'poly'}          F1 Score: 0.6646
# {'svm__C': 1000, 'svm__gamma': 0.1, 'svm__kernel': 'rbf'}          F1 Score: 0.6574


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

# create interactive wordcloud
# interactive_wordcloud(X)


# ======================================================================================================================
# Train/Test set split
# ======================================================================================================================

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=0)


# ======================================================================================================================
# Tune weight for imbalanced classification in XGB Classifier
# ======================================================================================================================

# class_weights = list(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))
# 0     1430
# 1    19189
# 2     4160

# count of the labels
class_count = Counter(y_train)

# the max label count
all_values = class_count.values()
max_count = max(all_values)

# undervalue class 2 ('neither', 3/4 of the majority class)
class_weights = {'hate speech': max_count / class_count['hate speech'],
                 'offensive language': max_count / class_count['offensive language'],
                 'neither': 3/4 * (max_count / class_count['neither'])}
# [5.775964775964776, 0.4304381393553368, 1.9854878917378918]
print(class_weights)
print(y_train.shape[0])

# assign the corresponding class weight for each individual data instance
w_array = np.ones(y_train.shape[0], dtype='float')
for i, val in enumerate(y_train):
    w_array[i] = class_weights[val]

print(w_array)
print(y_train)

# ======================================================================================================================

# Train model
model.fit(x_train, y_train, xgb__sample_weight=w_array)

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
print("Balanced Accuracy: %.2f" % balanced_accuracy_score(y_test, y_predicted))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))


# ======================================================================================================================
# AFTER FITTING THE MODEL, SAVE IT WITH PICKLE
# ======================================================================================================================
'''
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
# [5.775964775964776, 0.4304381393553368, 1.9854878917378918]
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

# Saving model to disk
pickle.dump(model, open('model.pkl', 'wb'))
'''

'''
# Load the model to make predictions
model = pickle.load(open('model.pkl','rb'))
y_predicted = model.predict(x_test)
'''