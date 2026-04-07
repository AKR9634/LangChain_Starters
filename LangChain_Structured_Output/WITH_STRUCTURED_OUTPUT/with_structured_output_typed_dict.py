from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

# class Review(TypedDict):

#     summary: str
#     sentiment: str

# Can also use Annotated 
class Review(TypedDict):

    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review... as a list"]
    summary: Annotated[str, "A brief summary of the review."]
    sentiment: Annotated[str, "Return the sentiment of the review"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside the list"]


llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")

model = ChatHuggingFace(llm=llm)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("Hey, The hardware is great, but the software feels bloated. There are too many pre-installed apps that i cant remove. Also, the UI  looks outdated compared to other brands. Hoping for a software update to fix this.")

print(result)