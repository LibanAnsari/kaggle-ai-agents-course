from dotenv import load_dotenv

import uuid
load_dotenv()

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
print("✅ ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

# MCP integration with Everything Server
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)
print("✅ MCP Tool created")


# Create image agent with MCP integration
image_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction="Use the MCP Tool to generate images for user queries",
    tools=[mcp_image_server],
)

async def main():
    runner = InMemoryRunner(agent=image_agent)

    response = await runner.run_debug("Provide a sample tiny image", verbose=True)

    from IPython.display import display, Image as IPImage
    import base64

    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    for item in part.function_response.response.get("content", []):
                        if item.get("type") == "image":
                            # Decode the base64 image bytes
                            img_bytes = base64.b64decode(item["data"])
                            # Try inline display (works in Jupyter/IPython environments)
                            try:
                                display(IPImage(data=img_bytes))
                            except Exception:
                                pass
                            # Persist image to disk so it can be opened normally
                            import os, datetime
                            out_dir = os.path.join(os.path.dirname(__file__), "generated_images")
                            os.makedirs(out_dir, exist_ok=True)
                            file_name = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            file_path = os.path.join(out_dir, file_name)
                            with open(file_path, "wb") as f:
                                f.write(img_bytes)
                            print(f"Saved image to: {file_path}")
                            
import asyncio
if __name__ == "__main__":
    asyncio.run(main())