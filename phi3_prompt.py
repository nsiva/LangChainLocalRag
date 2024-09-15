from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="phi3:mini" #"llama3.1:8b",
)


prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model
response = chain.invoke({"topic":"bears"})
print(response)

print("content of response")
print(response.content)
