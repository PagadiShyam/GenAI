from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline  # Import Hugging Face pipeline

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
      
        # create embeddings using Hugging Face model
        embeddings = HuggingFaceEmbeddings(model_name="impira/layoutlm-document-qa")
        knowledge_base = FAISS.from_texts(chunks, embeddings)
      
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use Hugging Face Falcon model for question answering
            falcon_model = "impira/layoutlm-document-qa"  # Replace with your desired Hugging Face Falcon model
            qa_pipeline = pipeline("document-question-answering", model=falcon_model, tokenizer=falcon_model)
            
            response = []
            with get_openai_callback() as cb:
                for doc in docs:
                    result = qa_pipeline(question=user_question, context=doc)
                    response.append(result)
                print(cb)
                
            st.write(response)
    
if __name__ == '__main__':
    main()
