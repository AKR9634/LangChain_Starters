from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# text = "Delhi is the capital of India"
document = [
    "Delhi is the capital of India",
    "Paris is the capital of France", 
    "Dehradun is the capital of Uttarakhand"
]


# vector = embedding.embed_query(text)
vector = embedding.embed_documents(document)

print(str(vector))



# from sentence_transformers import SentenceTransformer
# sentences = ["Virat Kohli is a Cricketer"]

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# embeddings = model.encode(sentences)

# print(embeddings)