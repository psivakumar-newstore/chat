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

# -----------------------------
# 1. Load API Key from .env
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

# -----------------------------
# 2. Load Vector DB (FAISS)
# -----------------------------
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.load_local("vector_db_faiss", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# 3. Setup Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# -----------------------------
# 4. Retrieval Chain (to use docs)
# -----------------------------
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key),
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# -----------------------------
# 5. API Call Utility
# -----------------------------
def call_api(method, url, params=None, data=None, headers=None):
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

# -----------------------------
# 6. Tool: Wrapper for GPT to call the API
# -----------------------------
def eval_api_call(input_str):
    try:
        parsed = json.loads(input_str)
        return call_api(
            method=parsed.get("method"),
            url=parsed.get("url"),
            headers=parsed.get("headers"),
            params=parsed.get("params"),
            data=parsed.get("data")
        )
    except Exception as e:
        return {"error": f"Failed to parse input or call API: {str(e)}"}

api_tool = Tool(
    name="CallDynamicAPI",
    func=eval_api_call,
    description="""
    Use this to call external APIs based on documentation instructions.
    Input must be a JSON string like:
    {
        "method": "GET",
        "url": "https://api.example.com/items",
        "params": {"id": "123"},
        "headers": {"Authorization": "Bearer xyz"},
        "data": {"key": "value"}  # for POST/PUT
    }
    """
)

# -----------------------------
# 7. Tool: Document Retriever
# -----------------------------
doc_tool = Tool(
    name="DocsRetriever",
    func=lambda q: retrieval_chain.invoke({"question": q})["answer"],
    description="Use this to look up how to use an API from documentation."
)

# -----------------------------
# 8. Initialize Agent with Tools
# -----------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)

agent = initialize_agent(
    tools=[doc_tool, api_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# -----------------------------
# 9. Ask a Question to Trigger Retrieval or API
# -----------------------------
if __name__ == "__main__":
    print("\n🤖 GPT-4o API-Enabled Agent Ready!\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            result = agent.run(user_input)
            print("\nAssistant:", result)
        except Exception as err:
            print("❌ Error:", err)