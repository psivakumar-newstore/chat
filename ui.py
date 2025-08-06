import os
import json
import requests
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chains import ConversationalRetrievalChain

# --- Load environment ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file")
    st.stop()

# --- Embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# --- Load vector DB ---
db_path = Path("vector_db_chroma")
if not db_path.exists():
    st.error("❌ No Chroma DB found. Run load_docs_to_vector_db.py first.")
    st.stop()

vectorstore = Chroma(persist_directory="vector_db_chroma", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
st.success("✅ Using Chroma vector DB")

# --- Memory ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# --- Prompt for retrieval grounding ---
QA_TEMPLATE = """You are a helpful assistant.
Use only the following context to answer the question.
If the answer cannot be found, say: "The documents do not contain enough information."

Context:
{context}

Question: {question}

Answer:"""

custom_prompt = PromptTemplate.from_template(QA_TEMPLATE)

# --- Retrieval Chain (used as a tool) ---
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key),
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer",
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# --- Tool 1: Retrieval Tool ---
retriever_tool = Tool(
    name="DocsRetriever",
    func=lambda q: retrieval_chain.invoke({"question": q})["answer"],
    description="Use this to retrieve answers from the document database."
)

# --- Tool 2: API Calling Tool ---
def call_api(method, url, params=None, data=None, headers=None):
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            json=data,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def eval_api_call(json_input_str):
    try:
        parsed = json.loads(json_input_str)
        return call_api(
            method=parsed.get("method"),
            url=parsed.get("url"),
            headers=parsed.get("headers"),
            params=parsed.get("params"),
            data=parsed.get("data")
        )
    except Exception as e:
        return {"error": f"Parsing or API call failed: {str(e)}"}

api_tool = Tool(
    name="CallAPI",
    func=eval_api_call,
    description="""
    Use this tool to call an external API based on the documentation.
    Input must be a JSON string like:
    {
        "method": "GET",
        "url": "https://api.example.com/endpoint",
        "params": {"id": "123"},
        "headers": {"Authorization": "Bearer TOKEN"},
        "data": {"key": "value"}
    }
    """
)

# --- Agent: Use both tools ---
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)

agent = initialize_agent(
    tools=[retriever_tool, api_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# --- Streamlit UI ---
st.title("🤖 GPT-4o Chatbot + API Agent")
st.write("Ask questions about your docs, or let the bot call a real API if described.")

user_question = st.chat_input("Type your question (or API action)...")

if user_question:
    response = agent.run(user_question)
    st.session_state.setdefault("chat_history", [])
    st.session_state.chat_history.append({"user": user_question, "bot": str(response)})

# --- Display chat history ---
if "chat_history" in st.session_state:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])

# --- Optional debug sidebar ---
with st.sidebar:
    st.write("### 🛠️ Agent Tools")
    st.write("- DocsRetriever (from Chroma)")
    st.write("- CallAPI (dynamic REST API caller)")

    st.write("### 🔍 Tip")
    st.markdown("Ask things like:")
    st.code("Call the gift card balance API for card 12345", language="markdown")