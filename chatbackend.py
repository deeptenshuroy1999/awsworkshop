# 1->Import langchain functions
from langchain_aws import ChatBedrock
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

import transformers


#2 -> Create function to invoke model
def titan_llm():
    llm=ChatBedrock(
    model_id="amazon.titan-text-express-v1",
    model_kwargs={"temperature":0.5,
    "maxTokenCount":100,
    "topP":0.5}
    )
    return llm

#test llm:
    #return llm.invoke(input_text)
#response= titan_llm("Who is Subhankar Halder lgbtq?")
#print(response)
#3 3->Create memory functions for the chatbot
def create_memory():
    llm=titan_llm()
    memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=256
    )
    return memory
#4 4->Create a chat client function to run the chatbot
def get_chat_response(input_text,memory):
    llm=titan_llm()
    chat=ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    chat_response=chat.invoke(input=input_text)
    return chat_response['response']