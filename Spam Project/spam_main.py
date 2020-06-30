import streamlit as st

import spam_vizualitation as d
import spam_predict as mp



def sidebar():
    
    st.sidebar.title("About")
    st.sidebar.info(
        "This an open source project. " 
        "This app is maintained by **Rohan Gupta**. " 
        "Go check out my [Github account](https://github.com/rohan300557) :grinning: ")


st.title('Spam-Ham Project')
st.image("data/message.jpg", use_column_width = True)
sel = st.selectbox('Select from below',['--Select One--','All','Data Analysis','Predictor'])
if(sel=='--Select One--'):
    sidebar()
if(sel == 'Data Analysis'):
    d.main()
    sidebar()
if(sel == 'Predictor'):
    mp.main()
    sidebar()
if(sel == 'All'):
    d.main()
    mp.main()
    sidebar()
