from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

model = ChatOllama(
    model="phi3:mini" #"llama3.1:8b",
)


prompt = ChatPromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



storage_path="./chroma_langchain_db"

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=storage_path, embedding_function=local_embeddings)

chain = {"docs": format_docs} | prompt | model | StrOutputParser()



question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print("number of docs after similarity search")
print(len(docs))
print("content of first doc")
print(docs[0])

chain.invoke(docs)
