from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

str_parser = StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

pyd_parser = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative...:  \n {feedback} \n {format_instructions} ",
    input_variables=["feedback"],
    partial_variables={'format_instructions':pyd_parser.get_format_instructions()}                      
)

classifier_chain = prompt1 | model | pyd_parser

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x: "Could not find the sentiment!!!")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':"This is a defective phone!!!"}))

chain.get_graph().print_ascii()