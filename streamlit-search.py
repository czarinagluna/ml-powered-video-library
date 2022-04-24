import streamlit as st
import pandas as pd
import spacy
from tqdm import tqdm
from rank_bm25 import BM25Okapi

data = pd.read_csv('data/data.csv')

nlp = spacy.load('en_core_web_sm')
tokenized_text = [] 

for doc in tqdm(nlp.pipe(data['preprocessed_text'].fillna('').str.lower().values, disable=['tagger', 'parser', 'ner'])):
    tokenized = [token.text for token in doc if token.is_alpha]
    tokenized_text.append(tokenized)

bm25 = BM25Okapi(tokenized_text)

def search_video(query, result=3):
    tok_query = query.lower().split(' ')
    results = bm25.get_top_n(tok_query, data['file_path'], result)
    
    results_list = [video for video in results]
    return results_list

def show_search_page():
    st.image('data/images/header.png')

    st.title('Searchable Video Library')

    search_word = st.text_input('Search for a video!', '')

    if search_word != '':
        result_video = search_video(search_word)
        first_video = result_video[0]
        second_video = result_video[1]
        third_video = result_video[2]
        st.video(first_video)
        st.video(second_video)
        st.video(third_video)
