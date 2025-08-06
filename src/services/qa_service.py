from langgraph.graph import START, StateGraph
from config.config import *
from core.vector_store import get_vectorstore
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from db.db import get_chat_history, insert_application_logs, create_application_logs, get_db_connection
from langchain.chains import create_history_aware_retriever
from langchain import hub
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    
class QAService:
    def __init__(self):
        self.llm = init_chat_model(CHAT_MODEL_NAME, model_provider="google_genai")

        # create a retriever
        self.vectorstore = get_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs = {"k": SEARCH_COUNT})

        # create a history aware retriever
        contextualize_q_system_prompt = """
        Given a chat history and the latest user question
        which might reference context in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is.
        """

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("system", "context: {context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # self.prompt = hub.pull("rlm/rag-prompt")
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Explain step by step if being asked."
            "\n\n"
            "{context}"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
                ("system", "context: {context}"),
                MessagesPlaceholder("chat_history"),
            ]
        )
        
        print(type(self.prompt))
    
        self.chat_history = []

        # Application steps:
        def retrieve(state: State):
            retriever_result = self.history_aware_retriever.invoke(
                {"input" : state["question"], "context" : "", "chat_history" : self.chat_history}
            )
            return {"context":retriever_result}

        def generate(state: State):
            docs_content = "/n/n".join(doc.page_content for doc in state["context"])
            messages = self.prompt.invoke({
                "context":docs_content,
                "question": state["question"],
                "chat_history" : self.chat_history
            })
            answer = self.llm.invoke(messages)
            return {"answer":answer}

        def save_history(state: State):
            self.chat_history.append({
                "role": "human",
                "content": state["question"]
            })
            self.chat_history.append({
                "role" : "ai",
                "content": state["answer"].content
            })
    
        graph_builder = StateGraph(State).add_sequence([retrieve, generate, save_history])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()



    def get_chat_response(self, question : str, session_id : str):
        if session_id is not None:
            self.chat_history = get_chat_history(session_id)
        
        answer = self.graph.invoke({"question": question})['answer']
        insert_application_logs(session_id, question, answer.content, CHAT_MODEL_NAME)
        
        return answer.content
    
