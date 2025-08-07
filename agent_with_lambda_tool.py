import os
import json
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# ----------------------------
# 1. Load environment
# ----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
lambda_url = os.getenv("GPT_PROXY_LAMBDA_URL")

if not api_key or not lambda_url:
    raise ValueError("❌ Set both OPENAI_API_KEY and GPT_PROXY_LAMBDA_URL in .env")

# ----------------------------
# 2. Setup vector DB
# ----------------------------
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = Chroma(
    persist_directory="vector_db_chroma",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ----------------------------
# 3. Memory
# ----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# ----------------------------
# 4. ConversationalRetrievalChain
# ----------------------------
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key),
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

retriever_tool = Tool(
    name="DocsRetriever",
    func=lambda q: retrieval_chain.invoke({"question": q})["answer"],
    description="Use this tool to look up how to use an API from documentation."
)

# ----------------------------
# 5. Lambda-based API calling tool
# ----------------------------
def call_lambda_proxy(json_input):
    try:
        payload = json.loads(json_input)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": os.getenv("GPT_CALLER_API_KEY", "")  # optional header auth
        }
        response = requests.post(
            lambda_url,
            method='GET',
            headers=headers,
            data=json.dumps(payload),
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

api_tool = Tool(
    name="CallProxyAPI",
    func=call_lambda_proxy,
    description="""
Use this tool to make external API calls via the GPT Proxy Lambda.
Input should be a JSON string like:
{
  "method": "POST",
  "url": "https://api.example.com/endpoint",
  "headers": {"Content-Type": "application/json"},
  "payload": {"key": "value"}
}
"""
)

# ----------------------------
# 6. Agent with Tools
# ----------------------------
agent = initialize_agent(
    tools=[retriever_tool, api_tool],
    llm=ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key),
    memory=memory,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

# ----------------------------
# 7. CLI interface (can be imported into UI too)
# ----------------------------
if __name__ == "__main__":
    print("\n🤖 GPT Agent with Lambda Tool Ready!\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print("\nAssistant:", response)