import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import eli5
from eli5.lime import TextExplainer
import clean_data.preprocessing
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


# http://localhost:5000


# app = Flask(__name__, template_folder='template') # to rename the templates folder that have the index.html file
app = Flask(__name__)


# Used in pickle pipeline on TF-IDF
def dummy(token):
    return token


# Load pre-trained ML model
model = pickle.load(open('model.pkl', 'rb'))

# Create object of class preprocessing to clean data
reading = clean_data.preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True, only_verbs_nouns=False)

# ======================================================================================================================
# Pre-processing
# ======================================================================================================================
def preprocessing(sentence):
    regular1 = re.compile('!* *RT.*:')
    regular2 = re.compile('&[^;]*;')
    regular3 = re.compile('//t.co[^ ]*')
    # Delete URLs
    regular4 = re.compile('http\S+|www.\S+')
    # Delete Usernames
    regular5 = re.compile(r'@\S+')
    # Replace hashtags with space to deal with the case where the tweet appears to be one word but is consisted by more seperated from hashtags
    regular6 = re.compile(r'#\S+')

    temp = regular1.sub('', sentence[0])
    temp = regular2.sub('', temp)
    temp = regular3.sub('', temp)
    temp = regular4.sub('', temp)
    temp = regular5.sub('', temp)
    temp = regular6.sub(' ', temp)

    # Convert to lower case
    temp = temp.lower()

    # substitute contractions with full words
    temp = replace_contractions(temp)

    # Tokenize tweets
    temp = word_tokenize(temp)

    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    temp = [w.translate(table) for w in temp]

    # remove all tokens that are not alphabetic
    temp = [word for word in temp if word.isalpha()]

    # stemming of words
    porter = PorterStemmer()
    temp = [porter.stem(word) for word in temp]

    # Delete Stop-Words
    whitelist = ["n't", "not", 'nor', "nt"]  # Keep the words "n't" and "not", 'nor' and "nt"
    stop_words = set(list(stopwords.words('english')) + ['"', '|'])
    temp = [w for w in temp if w not in stop_words or w in whitelist]

    # join each word to a sentence in order to use default tokenizer and preprocessor of TfidfVectorizer
    # (due to error while loading pickle, using custom and lambda functions for tokenizer and preprocessor)
    sentence = ' '.join(word for word in temp)

    return [sentence]


# ======================================================================================================================
# Remove Contractions (pre-processing)
# ======================================================================================================================

def get_contractions():
    contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                        "could've": "could have", "couldn't": "could not", "didn't": "did not",
                        "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                        "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                        "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                        "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                        "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                        "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                        "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                        "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                        "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                        "she'll've": "she will have", "she's": "she is", "should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                        "so's": "so as", "this's": "this is", "that'd": "that would",
                        "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                        "there'd've": "there would have", "there's": "there is", "here's": "here is",
                        "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                        "they'll've": "they will have", "they're": "they are", "they've": "they have",
                        "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                        "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                        "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                        "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                        "when've": "when have", "where'd": "where did", "where's": "where is",
                        "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                        "will've": "will have", "won't": "will not", "won't've": "will not have",
                        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                        "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                        "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                        "you're": "you are", "you've": "you have"}

    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

def replace_contractions(text):
    contractions, contractions_re = get_contractions()

    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)



# ======================================================================================================================
# ADD ROUTES TO CREATE API
# ======================================================================================================================
@app.route('/')
def home():
    return render_template('index.html')


# EXAMPLE 1
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    form_text = [request.form.get('hate_speech_text_field')]
    # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

    if len(form_text): # page not reloaded, form_text array not empty [fix the bug of page reloading -  it return no values from forms]
        # preprocessing of the sentence about to predict
        final_features = reading.clean_text(form_text[0])

        print(final_features)
        '''
        final_features = preprocessing(form_text)
        print(final_features)
        '''

        prediction = model.predict(final_features)

        if prediction == 0:
            output = "hate speech"
        elif prediction == 1:
            output = "offensive language"
        else:
            output = "neither"


        # ==============================================================================================================
        # Explain prediction
        # ==============================================================================================================

        def predict_complex(docs):
            # preprocess the input doc
            print('hereee', docs)
            final_features = reading.clean_text(docs)
            print(final_features)
            y_preds = model.predict_proba(final_features)
            return y_preds


        # clf: define ML classifier
        # vec: define vectorizer
        te = TextExplainer(random_state=42)  # use LIME method to train a white box classifier to make the same prediction as the black box one (pipeline)
        # predict_proba: Black-box classification pipeline. predict_proba should be a function which takes a list of
        #                strings (documents) and return a matrix of shape (n_samples, n_classes)
        print('form_text[0]', form_text[0])
        test= te.fit(form_text[0], predict_complex)
        print(test)

        #te.fit(form_text[0], model.predict_proba)  # form_text[0]
        test134 = te.show_prediction(targets=[output], target_names=["hate speech", "offensive language", "neither"])
      #  eli5.format_as_html(te.explain_prediction(target_names=["hate speech", "offensive language", "neither"]))
        #print(eli5.format_as_text(te.show_prediction(target_names=["hate speech", "offensive language", "neither"])))

        # show how close the results of the white box classifier are compared to the black box one (pipeline)
        print(te.metrics_)  # mean_KL_divergence -> small (0%), score -> big (100%)

        # ==============================================================================================================

        '''
        # RETURNS HTML
        
        #test1 = eli5.show_prediction(model, doc=form_text[0])
        
        test1 = eli5.show_prediction(model.named_steps['svm'],
                     model.named_steps['tfidf'].transform(form_text))
                     
        print("here", test1)
        
        # ==============================================================================================================
        
        # RETURNS HTML, BUT NEEDS TO BE PASSED TROUGH MARK-UP TO BE SHOWN ON HTML WEB PAGE 
        
        explain_html = eli5.format_as_html(eli5.explain_prediction(model, doc=form_text[0]))
        print("AND HERE", explain_html)
        
        #print(eli5.format_as_image(eli5.explain_prediction(model, doc=form_text[0])))
        # format_as_image()
        
        # Write HTML String to file.html
        with open("explain_prediction.html", "w") as file:
            file.write(explain_html)

        from flask import Markup
        # pass html code from FLASK to HTML template
        explain_html = Markup(explain_html)

        '''

        return render_template('index.html', hate_speech_text="Hate Speech / Offensive Language Prediction:",
                               pre_predict_text="'"+form_text[0]+"'"+" is ", prediction_text=output,
#                               tesst=test1,
                               #test2=explain_html,
                               test134=test134)
    else:  # if page is reloaded the form_text array will be empty
        return render_template('index.html')



# EXAMPLE 2
@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        #write your function that loads the model
        year = request.form['year']
        predicted_stock_price = model.predict(year)
        return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(predicted_stock_price))



if __name__ == "__main__":
    app.run(debug=True)
    # app.run("localhost", "9999", debug=True)