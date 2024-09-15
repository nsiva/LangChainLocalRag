from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
print("number of documents after split")

print(len(all_splits))


local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings, persist_directory="./chroma_langchain_db")

question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print("number of docs after similarity search")
print(len(docs))
print("content of first doc")
print(docs[0])

