import os
import streamlit as st
from dotenv import load_dotenv
from app.utils import extract_text_from_pdfs
from app.rag_pipeline import get_text_chunks, build_vectorstore, answer_query

load_dotenv()

st.set_page_config(page_title="DocuQuery AI", page_icon="📄", layout="wide")

st.title("📄 DocuQuery AI — Multi-PDF RAG Chatbot")
st.markdown("Ask intelligent questions across uploaded PDF documents using Gemini + LangChain + FAISS.")

with st.sidebar:
    st.header("Upload Documents")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    process = st.button("Process PDFs")

    if process:
        if not pdf_docs:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Reading, chunking, and indexing your PDFs..."):
                raw_text = extract_text_from_pdfs(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                build_vectorstore(text_chunks)
            st.success("PDFs processed successfully. You can now ask questions.")

user_question = st.text_input("Ask a question about your uploaded documents")

if user_question:
    with st.spinner("Retrieving answer..."):
        try:
            response = answer_query(user_question)
            st.subheader("Answer")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure you processed the PDFs first and configured GOOGLE_API_KEY in .env")

st.markdown("---")
st.caption("Built by Smeet Patel | Streamlit + LangChain + FAISS + Gemini")
