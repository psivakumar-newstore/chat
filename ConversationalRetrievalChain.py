import os
import json
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chains import ConversationalRetrievalChain

# --- 1. Load environment ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found in .env file")

# --- 2. Load vectorstore ---
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.load_local("vector_db_faiss", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --- 3. Setup memory ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# --- 4. Define retrieval chain ---
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key),
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# --- 5. Dynamic API Caller ---
def call_dynamic_api(method: str, url: str, headers=None, params=None, data=None):
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def eval_api_call(input_str):
    try:
        parsed = json.loads(input_str)
        return call_dynamic_api(
            method=parsed.get("method"),
            url=parsed.get("url"),
            headers=parsed.get("headers"),
            params=parsed.get("params"),
            data=parsed.get("data")
        )
    except Exception as e:
        return {"error": str(e)}

# --- 6. Define Tools ---
tools = [
    Tool(
        name="RetrieveDocs",
        func=lambda q: retrieval_chain.invoke({"question": q})["answer"],
        description="Use this to answer questions using document context."
    ),
    Tool(
        name="CallDynamicAPI",
        func=eval_api_call,
        description="""
            Use this to call an external API. Input must be a JSON string like:
            {
                "method": "GET",
                "url": "https://api.example.com/endpoint",
                "headers": {"Authorization": "Bearer TOKEN"},
                "params": {"query": "something"},
                "data": {"key": "value"}
            }
        """
    )
]

# --- 7. Initialize Agent ---
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# --- 8. Run example query ---
query = "Call the gift card balance API using the documented method"
response = agent.run(query)

print("\n💬 Assistant:", response)