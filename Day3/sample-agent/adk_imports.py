# Environment
from dotenv import load_dotenv
load_dotenv()

# Standard libs
import asyncio
from typing import Any, Dict

# Heavy ADK & Google imports
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
