from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

model = ChatOllama(
    model="phi3:mini" #"llama3.1:8b",
)


RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    #return {'1':1, 'two':2}



storage_path="./chroma_langchain_db"

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=storage_path, embedding_function=local_embeddings)

retriever = vectorstore.as_retriever()

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

'''
chain = (
    RunnablePassthrough.assign(context=lambda inputs: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)
'''

question = "What are the approaches to Task Decomposition?"
print("Question -> " + question)

docs = vectorstore.similarity_search(question)
print("number of docs after similarity search")
print(len(docs))

print("content of first doc")
print(docs[0])

#chain_doc = chain.invoke(docs)
chain_doc = chain.invoke(question) #{"context": docs, "question": question})

print("content of qa chain_doc")

print(chain_doc)

print("metadata for first doc")

print(chain_doc[0].metadata)
