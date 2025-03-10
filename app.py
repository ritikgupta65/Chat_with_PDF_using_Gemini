import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
genai.configure(api_key=os.getenv('Google_API_KEY'))

def get_pdf_text(pdf_file):
    text = ""
    for pdf in pdf_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context then just say, "answer is not present in the context", don't provide the wrong answer.
    Context:\n {context}?\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def chatbot():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.title("📄 Chat with Multiple PDFs using Gemini")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files and click on Submit & Process", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector(text_chunks)
                st.success("Processing Complete ✅")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_question = st.chat_input("Ask a question from the PDF files...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        reply = response['output_text']
        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

if __name__ == "__main__":
    chatbot()
