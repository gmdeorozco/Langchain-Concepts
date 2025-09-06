import asyncio
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner  # Correct import
from google.genai import types  # ADK's Content and Part structures


joke_teller = Agent(
    name="joke_agent",
    model = "gemini-2.0-flash",
    description =(
        "Joke teller"
    ),
    instruction="You will tell a funny joke about the topic provided by the user. Ask the use for the topic",
    
)

root_agent = joke_teller