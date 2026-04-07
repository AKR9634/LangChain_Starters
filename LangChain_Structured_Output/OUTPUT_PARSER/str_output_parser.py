from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)


# Using the result.content instead of string output parser...
prompt1 = template1.invoke({'topic':'Black Hole'})
result = model.invoke(prompt1)
prompt2 = template2.invoke({'text', result.content})

final_result = model.invoke(prompt2)

print("\n\n\n# Using the result.content instead of string output parser...")
print(final_result.content)

# Using the string output parser...

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'virat kohli'})

print("\n\n\n# Using the string output parser...")
print(result)

