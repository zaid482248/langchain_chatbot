from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm  = ChatOllama(model="qwen2.5-coder:1.5b",
                  temperature=0.9 
                  )

# response = llm("What is the capital of France?")
# print(response.content)

# message = {
#     ("System", "You are an expert in AI"),
#     ("Human","What is RAG")
# }

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in {topic},give concise , accurate answer"),
        ("human","{question}")
    ]
)

chain = prompt | llm | StrOutputParser() # LCEL 

# response = chain.invoke({"topic":"AI", "question": "What is RAG?"})
# print(response.content)

for chunk in chain.stream({"topic":"AI", "question": "What is RAG?"}):
    print(chunk, end="", flush=True)


