from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_text(text)


def build_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("vectorstore")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible using the provided context only.
    If the answer is not in the context, say: 'The answer is not available in the uploaded documents.'

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def answer_query(user_question):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(user_question, k=4)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]
