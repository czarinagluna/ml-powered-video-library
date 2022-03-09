import streamlit as st
import pandas as pd
import spacy
from tqdm import tqdm
from rank_bm25 import BM25Okapi

df = pd.read_csv('tiktok_data.csv')

def search_video(query, n=3):
    nlp = spacy.load('en_core_web_sm')
    tok_text=[]

    for doc in tqdm(nlp.pipe(df['nlp_text'].values, disable=['tagger', 'parser', 'ner'])):
        tok = [t.text for t in doc if t.is_alpha]
        tok_text.append(tok)

    bm25 = BM25Okapi(tok_text)

    tok_query = query.lower().split(' ')
    results = bm25.get_top_n(tok_query, df['file_path'], n)
    
    results_list = [video for video in results]
    return results_list

def show_search_page():
    st.image('@czarinagluna.png')

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