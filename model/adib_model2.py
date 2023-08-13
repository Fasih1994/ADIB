from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
persist_directory = '/home/fasih/ADIB/chroma'

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
print(vectordb._collection.count())
llm_name = 'gpt-3.5-turbo'


def get_agent(chat_history:list=None):

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
    )

    if len(chat_history)>1:
        for user_message, ai_message in chat_history[:-1]:
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(ai_message)
    print(memory)
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name=llm_name, temperature=0),
        retriever=vectordb.as_retriever(),
        memory=memory
    )
    return qa

def ask_agent(agent:ConversationalRetrievalChain = None, query:str = None):
    result = agent({"question": query})
    return result