# Prerequisite libraries

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

def app(url,querry):
    loader = YoutubeLoader.from_youtube_url(youtube_url=url)
    transcript = loader.load()
    embeddings = HuggingFaceEmbeddings()
    rcts = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=20)
    docs = rcts.split_documents(transcript)

    db = FAISS.from_documents(docs,embeddings)

    docs = db.similarity_search(querry, k=3)
    docs_page_content = " ".join([d.page_content for d in docs])

    template = """
        hey you are a very helpful Ai assistant who is able to answer question about youtube videos based on the video's
        transcript: {source}
        Only use the factual imformation gathered from the transcript to answer the question also answer it in a very detailed manner more than 30 words.
        If you feel that you dont have enough imformation to answer the question say "I dont have enough imformation in order to answer this question".
        """

    llm = GoogleGenerativeAI(model="models/text-bison-001")

    system_msg_template = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question: {question}"
    human_msg_template = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_msg_template,human_msg_template],
    )


    chain = LLMChain(llm = llm,prompt = chat_prompt)

    responce = chain.run(question = querry,source = docs_page_content)
    
    return responce, docs
