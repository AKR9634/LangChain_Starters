from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="What is the one single advantage of the following device: \n {device}",
    input_variables=['device']
)

prompt2 = PromptTemplate(
    template="What is the one single disadvantage of the following device: \n {device}",
    input_variables=['device']
)

chain = RunnableParallel(
    {
        'adv': RunnableSequence(prompt1, model, parser),
        'disadv': RunnableSequence(prompt2, model, parser)
    }
)

print(chain.invoke({'device':'mobile'}))

