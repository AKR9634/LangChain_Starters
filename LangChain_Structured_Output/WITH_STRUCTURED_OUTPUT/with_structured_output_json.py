from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

json_schema = {
    "title":"Review",
    "type":"object",
    "properties":{
        "key_themes":{
            "type":"array",
            "items":{
                "type":"string"
            },
            "description":"Write down all the key themes discussed in the review in a list"
        },
        "summary":{
            "type":"string",
            "description":"A brief summary of the review"
        },
        "sentiment":{
            "type":"string",
            "enum":["pos", "neg"],
            "description":"Return the sentiment of the review"
        },
        "pros":{
            "type":["array", "null"],
            "items":{
                "type":"string"
            },
            "description":"Write down all the pros inside a list"
        }
    },
    "required":["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("Hey, The hardware is great, but the software feels bloated. There are too many pre-installed apps that i cant remove. Also, the UI  looks outdated compared to other brands. Hoping for a software update to fix this.")

print(result)