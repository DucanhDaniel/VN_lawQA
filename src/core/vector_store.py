# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from config.config import *
# from document_processor import DocumentProcessor

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config.config import *
from core.document_processor import DocumentProcessor
from langchain_huggingface import HuggingFaceEmbeddings

def get_vectorstore():
    document_processor = DocumentProcessor()
    splits = document_processor.get_splits()
    # create chroma vector store
#     embedding_function = GoogleGenerativeAIEmbeddings(model = EMBEDDING_MODEL_NAME)
    embedding_function = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    collection_name = COLLECTION_NAME
    vectorstore = Chroma.from_documents(
            collection_name = collection_name,
            documents = splits,
            embedding = embedding_function,
            persist_directory = "/chroma_db"
    )
    return vectorstore


# get_vectorstore()