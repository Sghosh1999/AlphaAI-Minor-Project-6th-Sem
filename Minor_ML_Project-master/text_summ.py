#Updated On 03/03
import streamlit as st
import time
import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from timeit import Timer
import spacy
from textblob import TextBlob
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Function for Text Analysis
def text_analysis(in_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(in_text)
    tokens = [('"Tokens":{} , "Lemma":{}'.format(token.text, token.lemma_)) for token in doc ]
    return tokens

# Function for Named Entities
def name_entity_analysis(in_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(in_text)
    tokens = [token.text for token in doc]
    entities = [(entity.text, entity.label_) for entity in doc.ents ]
    allData = [('"Tokens":{}, "Entities":{}'.format(tokens,entities))]
    return allData

# Function for Sentiment Analysis
def sentiment_analysis(in_text):
    blob = TextBlob(in_text)
    sentiment_result = blob.sentiment
    return sentiment_result[0]

# Function for Text Summarizer
# Using Gensim
def gensim_summ(in_text):
    summary = summarize(in_text)
    return summary

# Using Sumy
def sumy_summ(in_text, key):
    parser = PlaintextParser.from_string(in_text, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,key)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

# Using TF_IDF
def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable

def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) -> dict:
    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
    return sentenceValue


def _find_average_score(sentenceValue) -> int:

    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def run_summarization(text,th):
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    freq_matrix = _create_frequency_matrix(sentences)
    tf_matrix = _create_tf_matrix(freq_matrix)
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = _score_sentences(tf_idf_matrix)
    threshold = _find_average_score(sentence_scores)
    summary = _generate_summary(sentences, sentence_scores, th * threshold)
    return summary  


# Main Function
def main():
    st.title("File Content Analysis")
    st.subheader("Natural Language Processing")

    message = st.text_area("Enter the text below", "Enter here", key = 1)
        
    # Tokenization
    if st.checkbox("Tokens and Lemma"):
        st.subheader("Showing Tokens and Lemma")
        if st.button("Analyze",key = 14):
            nlp_result = text_analysis(message)
            with st.spinner("Waiting"):
                t = Timer(lambda : text_analysis(message))
                time.sleep(t.timeit(number=1))
                st.json(nlp_result)

    # Named Entity
    if st.checkbox("Named Entities"):
        st.subheader("Extract Named Entities")
        #message = st.text_area("Enter the text below", "Enter here", key = 2)
        if st.button("Extract",key = 13):
            nlp_result = name_entity_analysis(message)
            with st.spinner("Waiting"):
                t = Timer(lambda : name_entity_analysis(message))
                time.sleep(t.timeit(number=1))
                st.json(nlp_result)
    
    # Sentiment Analysis
    if st.checkbox("Sentiment Analysis"):
        st.subheader("Showing Your Sentiments")
        #message = st.text_area("Enter the text below", "Enter here", key = 3)
        if st.button("Analyze",key = 10):
            nlp_result = sentiment_analysis(message)
            with st.spinner("Waiting"):
                t = Timer(lambda : sentiment_analysis(message))
                time.sleep(t.timeit(number=1))
                if nlp_result > 0:
                    st.success("This is a positive reaction")
                elif nlp_result < 0:
                    st.error("This is a negative reaction")
                else:
                    st.warning("This is a neutral reaction")
    
    # Text Summarizaton
    if st.checkbox("Text Summarize"):
        st.subheader("Showing your Summarize text")
        #message = st.text_area("Enter the text below", "Enter here", key = 4)
        summr = st.selectbox("Choose the summarizer",("gensim","sumy","TF-IDF"))
        if summr == 'sumy':
            level = st.selectbox("Select the Summarization Level",("1","2","3","4"))
        if summr == 'TF-IDF':
            th = st.slider('Size of Summary', 0.0, 2.0, 0.1)
        if st.button("Summarize",key = 11):
            if summr == 'gensim':
                nlp_result = gensim_summ(message)
                st.info("Using Gensim")
                with st.spinner("Waiting"):
                    t = Timer(lambda : gensim_summ(message))
                    time.sleep(t.timeit(number=1))
                    st.success(nlp_result)
            elif summr == 'sumy':
                nlp_result = sumy_summ(message, level)
                st.info("Using Sumy")
                with st.spinner("Waiting"):
                    t = Timer(lambda : sumy_summ(message, level))
                    time.sleep(t.timeit(number=1))
                    st.success(nlp_result)
            elif summr == 'TF-IDF':
                nlp_result = run_summarization(message,th)
                with st.spinner("Waiting"):
                    t = Timer(lambda : run_summarization(message,th))
                    time.sleep(t.timeit(number=1))
                    st.success(nlp_result)



if __name__ == '__main__':
    main()



