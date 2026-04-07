from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="finexplore.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0
)

# chunks = splitter.split_text(text)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)
