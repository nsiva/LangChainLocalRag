from langchain_ollama import ChatOllama

model = ChatOllama(
    model="phi3:mini" #"llama3.1:8b",
)

print("after phi3 mini model construction")

question = "Tell me a joke about indian sardarji"

print("question ->" + question)
response_message = model.invoke(
	question
)

print("after invoke phi3 with question")

print(response_message.content)
