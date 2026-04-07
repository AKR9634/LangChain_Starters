from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
document = [
    "Delhi is the capital of India",
    "Paris is the capital of France", 
    "Dehradun is the capital of Uttarakhand"
]


# result = embedding.embed_query("Delhi is the Capital of India!!!")
result = embedding.embed_documents(document)

print(str(result))