import logging
import asyncio
import os

# Configure logging with DEBUG log level.
# logging.basicConfig(
#     filename="logger.log",
#     level=logging.DEBUG,
#     format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
#     # force=True # Always updates the logger file, if not True then does nothin
# )

# Configure logging with DEBUG log level.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
    handlers=[
        logging.FileHandler("logger.log"),
        logging.StreamHandler()
    ],
    force=True
)

print("✅ Logging configured")

from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search

from google.genai import types
from typing import List

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# ---- Intentionally pass incorrect datatype - `str` instead of `List[str]` ----
def count_papers(papers: List[str]):
    """
    This function counts the number of papers in a list of strings.
    Args:
      papers: A list of strings, where each string is a research paper.
    Returns:
      The number of papers in the list.
    """
    return len(papers)


# Google Search agent
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Searches for information using Google search",
    instruction="""Use the google_search tool to find information on the given topic. Return the raw search results.
    If the user asks for a list of papers, then give them the list of research papers you found and not the summary.""",
    tools=[google_search]
)


# Root agent
root_agent = LlmAgent(
    name="research_paper_finder_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Your task is to find research papers and count them. 

    You MUST ALWAYS follow these steps:
    1) Find research papers on the user provided topic using the 'google_search_agent'. 
    2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
    3) Return both the list of research papers and the total number of papers.
    """,
    tools=[AgentTool(agent=google_search_agent), count_papers]
)
print("✅ Agent created")

# adk web --log_level DEBUG


print("----- EXAMPLE PLUGIN - DOES NOTHING ----- ")

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import BaseTool
from google.adk.plugins.base_plugin import BasePlugin


# Applies to all agent and model calls
class CountInvocationPlugin(BasePlugin):
    """A custom plugin that counts agent and tool invocations."""

    def __init__(self) -> None:
        """Initialize the plugin with counters."""
        super().__init__(name="count_invocation")
        self.agent_count: int = 0
        self.tool_count: int = 0
        self.llm_request_count: int = 0

    # Callback 1: Runs before an agent is called. You can add any custom logic here.
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Count agent runs."""
        self.agent_count += 1
        logging.info(f"[Plugin] Agent run count: {self.agent_count}")

    # Callback 2: Runs before a model is called. You can add any custom logic here.
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Count LLM requests."""
        self.llm_request_count += 1
        logging.info(f"[Plugin] LLM request count: {self.llm_request_count}")
        
    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict, tool_context: ToolContext
    ) -> None:
        """Count tool calls."""
        self.tool_count += 1
        logging.info(f"[Plugin] Tool calls count: {self.tool_count}")
        
# Plugging provided by ADK
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import LoggingPlugin # <---- 1. Import the Plugin
from google.genai import types


runner = InMemoryRunner(
    agent=root_agent,
    plugins=[
        # LoggingPlugin(),
        CountInvocationPlugin()
    ],  # <---- 2. Add the plugin. Handles standard Observability logging across ALL agents
)

print("✅ Runner configured")

async def run():
    print("🚀 Running agent with LoggingPlugin...")
    print("📊 Watch the comprehensive logging output below:\n")
    
    response = await runner.run_debug("Find recent papers on quantum computing")

if __name__ == "__main__":
    asyncio.run(run())