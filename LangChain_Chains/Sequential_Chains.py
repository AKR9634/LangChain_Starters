from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Generate a detailed content on \n {topic} ",
    input_variables=["topic"]                      
)

template2 = PromptTemplate(
    template="""Generate a 5 line summary on the following text: \n {text}
    
    in this bullet format...

    1. .............
    2. .............
    3. .............
    4. .............
    5. .............

    """,
    input_variables=["text"]
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Virat Kohli'})

print(result)

chain.get_graph().print_ascii()

