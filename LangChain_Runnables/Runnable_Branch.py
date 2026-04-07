from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch
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
    template="Generate a paragraph on the following topic: \n {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a summary of the following text: \n {text}",
    input_variables=['text']
)


para_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
        (lambda x: len(x.split())>100, RunnableSequence(prompt2, model, parser)),
        RunnablePassthrough()
)

final_chain = RunnableSequence(para_chain, branch_chain)

print(final_chain.invoke({'topic':"Uttarakhand"}))






