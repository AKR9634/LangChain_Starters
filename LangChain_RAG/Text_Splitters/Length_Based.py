from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="finexplore.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    separator='',
    chunk_size = 100,
    chunk_overlap = 0
)

# result = splitter.split_text(docs)

result = splitter.split_documents(docs)

print(result[0].page_content)

