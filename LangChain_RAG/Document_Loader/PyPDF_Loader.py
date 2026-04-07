from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("finexplore.pdf")

docs = loader.load()

print(len(docs))

print(type(docs[0]))

print(docs[0].page_content)