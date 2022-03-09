# Streamlit app

import streamlit as st
from search_page import show_search_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Search or Explore", ("Search", "Explore"))

#add_page = st.sidebar.selectbox("Connect with Me", ("[LinkedIn](https://www.linkedin.com/in/czarinaluna/)", "[GitHub](https://github.com/czarinagluna)")


if page == "Search":
    show_search_page()
else:
    show_explore_page()