from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

document = [
    Document(page_content="LangChain helps developer build LLM applications easily..."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search..."),
    Document(page_content="Embeddings convert text into high dimensional vectors..."),
    Document(page_content="OpenAI provides powerful embedding models..."),
    Document(page_content="LangChain provides chains that allow combining multiple components into a single workflow..."),
    Document(page_content="Agents in LangChain enable LLMs to make decisions and use tools dynamically..."),
    Document(page_content="LangChain supports memory modules that allow LLMs to retain context across interactions..."),
    Document(page_content="Retrieval Augmented Generation (RAG) in LangChain combines vector search with LLMs for better answers..."),
    Document(page_content="LangChain integrates with multiple vector stores like FAISS, Pinecone, and Chroma..."),
    Document(page_content="Prompt templates in LangChain help standardize and reuse prompts efficiently..."),
    Document(page_content="LangChain Expression Language (LCEL) allows declarative composition of chains..."),
    Document(page_content="LangChain supports document loaders for ingesting PDFs, websites, and text files..."),
    Document(page_content="Text splitters in LangChain break large documents into smaller chunks for processing..."),
    Document(page_content="LangChain retrievers abstract the process of fetching relevant documents from vector stores..."),
    Document(page_content="Tools in LangChain allow LLMs to interact with external APIs and systems..."),
    Document(page_content="LangChain can be used to build chatbots, question answering systems, and knowledge assistants..."),
    Document(page_content="Streaming in LangChain enables real-time token generation from LLMs..."),
    Document(page_content="LangChain callbacks help monitor and debug LLM application execution..."),
    Document(page_content="LangChain supports multi-modal applications combining text, images, and other data types..."),
    Document(page_content="LangChain Hub provides a repository of reusable prompts, chains, and components..."),
    Document(page_content="LangChain integrates with popular LLM providers like OpenAI, Anthropic, and Hugging Face..."),
    Document(page_content="LangChain enables evaluation and testing of LLM pipelines for performance tuning..."),
    Document(page_content="Conversational chains in LangChain help maintain dialogue history for chat applications..."),
    Document(page_content="LangChain can orchestrate complex workflows involving multiple LLM calls and tools...")
]

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectorstore = FAISS.from_documents(
    documents=document,
    embedding=embedding_model
)

similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5}),
    llm = model
)

query = "How can we build Apps using LLMs?"

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)


for i, doc in enumerate(similarity_results):
    print(f"\n --- Result {i+1} ---")
    print(doc.page_content)

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"\n --- Result {i+1} ---")
    print(doc.page_content)

