import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Optional: fallback to Chroma if FAISS fails
try:
    from langchain.vectorstores import FAISS
    use_faiss = False
except ImportError:
    use_faiss = False

# 1. Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. Define folders to scan
folders = [
    "docs_com",  # replace with your first folder path
    "docs_net"   # replace with your second folder path
]

all_documents = []

# 3. Load documents from both folders
for folder in folders:
    if not os.path.exists(folder):
        print(f"⚠️ Skipping missing folder: {folder}")
        continue
    loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} docs from {folder}")
    all_documents.extend(docs)

print(f"Total documents loaded: {len(all_documents)}")

# 4. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(all_documents)
print(f"Total chunks after splitting: {len(chunks)}")

# 5. Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# 6. Store in Vector DB
if use_faiss:
    print("Using FAISS vector store")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vector_db_faiss")
    print("✅ Saved FAISS index to ./vector_db_faiss")
else:
    print("FAISS not available, falling back to Chroma")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="vector_db_chroma")
    vectorstore.persist()
    print("✅ Saved Chroma index to ./vector_db_chroma")