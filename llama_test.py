from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:8b",
)

print("after llama 8b model construction")

response_message = model.invoke(
    "Simulate a rap battle between Stephen Colbert and John Oliver"
)
print("after invoke llama 8b ")

print(response_message.content)
