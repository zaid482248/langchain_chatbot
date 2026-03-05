from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_core.output_parsers import StrOutputParser


llm  = ChatOllama(model="qwen2.5-coder:1.5b",
                  temperature=0.9 
                  )

# LLM - Model | feature
# LLM - Tools (Web-search , files-search)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in {topic},give concise , accurate answer"),
        MessagesPlaceholder(variable_name="Chat_history"),
        ("human","{question}")
    ]
)

chain = prompt | llm | StrOutputParser() # LCEL 

# response = chain.invoke({"topic":"AI", "question": "What is RAG?"})
# print(response)
chat_list = []
max_turns = 5

def ask_question(question):
    current_turns = len(chat_list) // 2  # Each turn consists of a human and an AI message
    if current_turns >= max_turns:
        return (
            "Context window is full. Please start a new conversation or clear the chat history."
        )
    

    response = chain.invoke({
        "topic" : "AI",
        "question": question,
        "Chat_history": chat_list  # send the chat history to the llm model for context
    })

    chat_list.append(HumanMessage(content=question))
    chat_list.append(AIMessage(content=response))

    remaining_turns = max_turns -(current_turns +1)
    if remaining_turns <= 2:
        print(f"warning: Only {remaining_turns} turns left before the context window is full.")

    return response
def main():
    print("Chatbot Ready! (type 'quit' to exit, 'clear' to reset memory)\n")

    while True:
        user_input = input("You:").strip()

        if  not user_input:
            continue
        if user_input.lower() == "quit":
            print("Exiting the chatbot. Goodbye!")
            break
        if user_input.lower() == "clear":
            chat_list.clear()
            print("Chat history cleared.")
            continue
        print(f"AI: {ask_question(' '.join(user_input))}")


# print(ask_question("What is RAG"))
# print(ask_question("Give me a python example of it"))
# print(ask_question("Now explainthe code you just gave me"))

main()


