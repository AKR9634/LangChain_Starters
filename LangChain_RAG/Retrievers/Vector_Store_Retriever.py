from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

document = [
    Document(page_content="LangChain helps developer build LLM applications easily..."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search..."),
    Document(page_content="Embeddings convert text into high dimensional vectors..."),
    Document(page_content="OpenAI provides powerful embedding models...")
]

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="Chorma_DB",
    collection_name="my_collection"
)

vectorstore.add_documents(document)

retriever = vectorstore.as_retriever(search_kargs={"k":2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n --- Result {i+1} ---")
    print(doc.page_content)

