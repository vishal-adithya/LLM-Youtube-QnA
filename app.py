import streamlit as st
import time
from main import *
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import YoutubeLoader
from langchain.chains import LLMChain
from langchain.prompts import (SystemMessagePromptTemplate,HumanMessagePromptTemplate,
                               ChatPromptTemplate)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

st.header("Youtube Q&A")

querry_input = st.chat_input()
url_input = st.text_input("Your Youtube url")

bot = st.chat_message("ai")
bot.write("Hello!,\n enter the url in the allocated area and type the query.")

if querry_input:
    user = st.chat_message("user")
    user.write(querry_input)
    bot = st.chat_message("ai")
    with bot.status("Patience and thee shalt knoweth thy answ'r") as status:
        st.write("Creating Faiss database...")
        time.sleep(1)
        st.write("Generating answer for your querry....")

        response,docs = app(url=url_input,querry=querry_input)
        status.update(label= "Done!",state="complete")
    bot = st.chat_message("ai")
    bot.write(response)

