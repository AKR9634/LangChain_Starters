from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a 3 line poem on the following topic: \n {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Suggest the best single catchy title in one single for the following poem:  \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

poem_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        'title': RunnableSequence(prompt2, model, parser),
        'poem': RunnablePassthrough()
    }
)

chain = RunnableSequence(poem_chain, parallel_chain)

print(chain.invoke({'topic':'Uttarakhand'}))
