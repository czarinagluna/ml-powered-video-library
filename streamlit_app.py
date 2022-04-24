import streamlit as st
from steamlit_search import show_search_page
from streamlit_explore import show_explore_page


###### Sidebar ######
st.sidebar.write('<h1 style="text-align:center">Machine Learning-Powered Video Library</h1>', unsafe_allow_html=True)
st.sidebar.write('<h5 style="text-align:center">By <a href="https://czarinaluna.com">Czarina Luna</a></h5>', unsafe_allow_html=True)

page = st.sidebar.selectbox("Search or Explore", ("Search", "Explore"))

st.sidebar.caption('Source Code on [Github](https://github.com/czarinagluna/ml-powered-video-library)')

st.sidebar.write('<h3>About</h3>', unsafe_allow_html=True)
st.sidebar.markdown('Video sharing applications today lack the functionality for users to search videos by their content. As a solution, I developed a searchable video library that processes videos and returns exact matches to queries using machine learning and artificial intelligence including speech recognition, optical character recognition, and object detection.')

st.sidebar.caption('Feel free to contact me for any questions and connect with me on [Linkedin](https://www.linkedin.com/in/czarinagluna/).')

###### Pages ######
if page == "Search":
    show_search_page()
else:
    show_explore_page()
