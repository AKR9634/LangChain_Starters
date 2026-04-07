from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#-------------------------------------------------------------------------------------------------------------

# Model Building

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

#-------------------------------------------------------------------------------------------------------------
# Transcript Generation

# https://www.youtube.com/watch?v=J2rQTJby8XM

video_id = "J2rQTJby8XM" # Not the full URL, just the video id tag!!!

yt_api = YouTubeTranscriptApi()

transcripts = yt_api.fetch(video_id=video_id, languages=['en'])

final_transcript = " ".join(item.text for item in transcripts)

#-------------------------------------------------------------------------------------------------------------

# Text Splitting

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=10)

docs = splitter.split_text(final_transcript)


#-------------------------------------------------------------------------------------------------------------

# Vector Store

vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings)

#-------------------------------------------------------------------------------------------------------------

# Retrievers

retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":5, "lambda_mult":0.2})

# for i, doc in enumerate(final_context):
#     print(f" Search {i+1} : \n")
#     print(doc.page_content)

#-------------------------------------------------------------------------------------------------------------

# Prompt Formation

question = "what are the steps involved in Git?"

similar_context = retriever.invoke(question)

final_context = "\n\n".join(item.page_content for item in similar_context)


prompt = PromptTemplate(
    template="""
    You are a great AI Assistant!!!
    I want you to use only the following context to answer the question...
    If context is insufficient then reply that you don't know the answer...

    Context:\n
    {context}

    Question:\n
    {question}
""",
input_variables=['context', 'question']
)

#-------------------------------------------------------------------------------------------------------------

chain = prompt | model | parser

result = chain.invoke({'context':final_context, 'question':question})

print(result)



