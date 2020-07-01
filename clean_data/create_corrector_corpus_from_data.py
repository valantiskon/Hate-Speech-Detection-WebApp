import pandas as pd
from pandas import read_csv

import clean_data.preprocessing
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================
# initialize preprocessing

reading = clean_data.preprocessing.preprocessing(use_spell_corrector=False)

# ======================================================================================================================
# Twitter data

# Read Twitter data
data = read_csv('../dataset/hate_tweets.csv')
#data = data.sample(n=1000, random_state=42)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.tweet.progress_map(reading.clean_text)

# ======================================================================================================================
# Instagram data

'''
# Read Instagram data
data = pd.read_csv("dataset/test_cleaned.csv", index_col=False)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.caption.progress_map(reading.clean_text)
'''

print(data)

# ======================================================================================================================
# Preprocessesing and tokenizer for CountVectorizer

def dummy(doc):
    return doc

# ======================================================================================================================
# Uni-grams
# ======================================================================================================================

# min_df: remove words occuring less than 0.2% of the posts
# max_df: remove words occuring more than a percentage of the posts (default 100%)
cv_unigrams = CountVectorizer(min_df=0.002, analyzer='word', ngram_range=(1, 1), lowercase=False, tokenizer=dummy, preprocessor=dummy)
cv_fit_unigrams = cv_unigrams.fit_transform(data['clean_text'])

vocabulary_unigrams = cv_unigrams.get_feature_names()
print(vocabulary_unigrams)

term_frequency_unigrams = cv_fit_unigrams.toarray().sum(axis=0)
print(term_frequency_unigrams)

vocab_freq_unigrams = {'vocab': vocabulary_unigrams, 'freq': term_frequency_unigrams}
df_unigrams = pd.DataFrame(vocab_freq_unigrams)
print(df_unigrams)


# ======================================================================================================================
# Bi-grams
# ======================================================================================================================

# min_df: remove words occuring less than 0.09% of the posts
# max_df: remove words occuring more than a percentage of the posts (default 100%)
cv_bigrams = CountVectorizer(min_df=0.0009, analyzer='word', ngram_range=(2, 2), lowercase=False, tokenizer=dummy, preprocessor=dummy)
cv_fit_bigrams = cv_bigrams.fit_transform(data['clean_text'])

vocabulary_bigrams = cv_bigrams.get_feature_names()
print(vocabulary_bigrams)

term_frequency_bigrams = cv_fit_bigrams.toarray().sum(axis=0)
print(term_frequency_bigrams)

vocab_freq_bigrams = {'vocab': vocabulary_bigrams, 'freq': term_frequency_bigrams}
df_bigrams = pd.DataFrame(vocab_freq_bigrams)
print(df_bigrams)


# ======================================================================================================================
# Store uni-gram and bi-gram dictionaries to csv
# ======================================================================================================================

'''
# save instagram cleaned data, from dataframe, to csv
df_unigrams.to_csv('../dataset/sym_spell-dictionaries/unigram_insta_posts_dict.csv', header=False, index=False, encoding='utf-8', sep=' ')
df_bigrams.to_csv('../dataset/sym_spell-dictionaries/bigram_insta_posts_dict.csv', header=False, index=False, encoding='utf-8', sep=' ')
'''

# save Twitter cleaned data, from dataframe, to csv
df_unigrams.to_csv('../dataset/sym_spell-dictionaries/unigram_twitter_posts_dict.csv', header=False, index=False, encoding='utf-8', sep=' ')
df_bigrams.to_csv('../dataset/sym_spell-dictionaries/bigram_twitter_posts_dict.csv', header=False, index=False, encoding='utf-8', sep=' ')

# ======================================================================================================================

print("Custom uni-grams: ", dict(zip(vocabulary_unigrams, term_frequency_unigrams)))
print("Custom bi-grams: ", dict(zip(vocabulary_bigrams, term_frequency_bigrams)))
