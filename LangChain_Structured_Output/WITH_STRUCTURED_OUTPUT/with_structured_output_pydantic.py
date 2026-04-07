from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review... as a list")
    summary: str = Field(description="A brief summary of the review.")
    sentiment: Literal["pos", "neg"] = Field(description="Return the sentiment of the review")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside the list")


structured_model = model.with_structured_output(Review)

result = model.invoke("Hey, The hardware is great, but the software feels bloated. There are too many pre-installed apps that i cant remove. Also, the UI  looks outdated compared to other brands. Hoping for a software update to fix this.")

print(result)