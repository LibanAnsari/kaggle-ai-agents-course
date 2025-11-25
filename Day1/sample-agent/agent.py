# adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY
# This intializes a folder named sample-agent containing a .env, init.py, and agent.py

import asyncio
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
print("✅ ADK components imported successfully.")

# agent automatically retries if something goes wrong
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.", # Can be optional ig, skipped in the part 2 lab of day1. 
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)
print("✅ Root Agent defined.")


runner = InMemoryRunner(agent=root_agent)
print("✅ Runner created.")

async def run_agent():
    response = await runner.run_debug(
        "What is a Runner in Agent Development Kit from Google?"
    )
    # Gets print automatically without the need of print statement
    
    # print(response) # Contains Metadata with response 
        
if __name__ == "__main__":
    asyncio.run(run_agent())