from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

API_KEY = "058e7445cd6d419ff49d81465952392e"
# URL = "http://api.weatherstack.com/current?access_key=058e7445cd6d419ff49d81465952392e&query=Mandi"

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This fucntion fetches the current weather data for a given city!!!
    """
    
    url = f"https://api.weatherstack.com/current?access_key={API_KEY}&query={city}"

    response = requests.get(url)

    return response.json()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = hub.pull("hwchase17/react")

# agent = create_agent(model=model, tools=[search_tool, get_weather_data])

agent = create_react_agent(llm=model, tools=[search_tool, get_weather_data], prompt=prompt)

agent_executor = AgentExecutor(agent = agent, tools=[search_tool, get_weather_data], verbose=True)

response = agent_executor.invoke({"input": "Find the capital of Uttrakhand, then find it's current weather condition!!!"})

print(response)


