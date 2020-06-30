import streamlit as st
import re
import nltk
import string 
import pandas as pd
import seaborn as se
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import plotly.express as px


dataset_loc = 'data/SMSSpamCollection'


def load_sidebar():
    st.sidebar.subheader("Spam ham Data Analysis")


@st.cache

def load_data(dataset_loc):
    df = pd.read_csv(dataset_loc, sep="\t",names = ['label','message'], error_bad_lines=False)
    return df

def load_description(df):
    st.header("Data Preview")
    if(st.checkbox("Top/Bottom rows of Dataset")):
        preview = st.radio("Choose one", ("Top", "Bottom"))
        if(preview == "Top"):
            st.write(df.head())
        if(preview == "Bottom"):
            st.write(df.tail())

    if(st.checkbox("Complete Dataset")):
        st.write(df)

    if(st.checkbox("Display the shape")):
        st.write(df.shape)
        dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
        if(dim == "Rows"):
            st.write("Number of Rows", df.shape[0])
        if(dim == "Columns"):
            st.write("Number of Columns", df.shape[1])

    if(st.checkbox('Counts of Unique values')):
        st.write(df.label.value_counts())

    if(st.checkbox('Describe using Group Labels')):
        st.write(df.groupby('label').describe())

def load_wordcloud(df, kind):
    words = ''
    for msg in df[df['label']==kind]['message']:
        msg = msg.lower()
        words += msg + ''
    wordcloud = WordCloud(width = 1600, height = 800).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    wordcloud.to_file("data/wc.png")
 
def load_viz(df):
    st.header("Data Visualisation")
    
    if(st.checkbox("Seaborn - Spam Ham Count")):
        plt.figure(figsize=(8,5))
        st.write(se.countplot('label',palette="Set2", data=df))
        st.pyplot()
    if(st.checkbox("Word Cloud")):
        preview = st.radio("Choose the sentiment?", ("spam", "ham"))
        st.write(preview,"Wordcloud:")
        load_wordcloud(df, preview)
        st.image("data/wc.png", use_column_width=True)

def main():
    load_sidebar()
    
    st.title('Spam-Ham Data Analysis')

    df = load_data(dataset_loc)

    load_description(df)

    load_viz(df)
if(__name__ == '__main__'):
    main()