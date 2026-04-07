from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('Virat.txt', encoding='utf-8')

docs = loader.load()

# print(type(docs))

# print(type(docs[0]))

# print(docs[0].page_content)

# print(docs[0].metadata)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Give me the summary of the following text in just 5 bullet points: \n {text}",
    input_variables=['text']
)

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))