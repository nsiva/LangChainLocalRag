from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

storage_path="./chroma_langchain_db"

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=storage_path, embedding_function=local_embeddings)




question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print("number of docs after similarity search")
print(len(docs))
print("content of first doc")
print(docs[0])

