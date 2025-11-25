import asyncio
import sqlite3
from typing import Any, Dict
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# from adk_imports import *
print("✅ ADK components imported successfully.")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"

# Define helper functions that will be reused throughout the notebook
async def run_session(runner_instance: Runner, user_queries: list[str] | str = None, session_name: str = "default", session_service = InMemorySessionService | DatabaseSessionService):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name
    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")
print("✅ Helper functions defined.")


async def InMemoryAgent():
    
    # Step 1: Create the LLM Agent
    root_agent = Agent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot",  # Description of the agent's purpose
    )

    # Step 2: Set up Session Management
    # InMemorySessionService stores conversations in RAM (temporary)
    session_service = InMemorySessionService()

    # Step 3: Create the Runner
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    print("✅ Stateful agent initialized!")
    print(f"   - Application: {APP_NAME}")
    print(f"   - User: {USER_ID}")
    print(f"   - Using: {session_service.__class__.__name__}")
    
    # Test the runner
    response = await run_session(
        runner, 
        [ # These are the set of new queries, any previous conversation will already be stored in memory under the session name
            "Hi, I am Sam! What is the capital of United States?",
            "Hello! What is my name?",  # This time, the agent should remember!
        ],
        "stateful-agentic-session-01",
        session_service
    )
    
    # Testing Agents forgetfulness (if ran this after the above queries, then the agent will retain the memories but if asked after the above queries are gone out of memory(i.e. after the program ended) the agent will not be able to answer)
    response = await run_session(
        runner, 
        [
            "What did I ask you about earlier?",
            "Hello, What is my name?"
        ],
        "stateful-agentic-session-01",
        session_service
    )
    response = await run_session(
        runner, 
        [
            "Hello, What is my name?"
        ],
        "stateful-agentic-session-02", # Changed session name (Agent does not know previous session events)
        session_service
    )


async def PersistentAgent():
    
    # Step 1: Create the same agent (notice we use LlmAgent this time)
    chatbot_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot with persistent memory",
    )

    # Step 2: Switch to DatabaseSessionService
    # SQLite database will be created automatically
    db_url = "sqlite:///Day3/sample-agent/my_agent_data.db"  # Local SQLite file
    session_service = DatabaseSessionService(db_url=db_url) # CHANGED !IMPORTANT

    # Step 3: Create a new runner with persistent storage
    runner = Runner(agent=chatbot_agent, app_name=APP_NAME, session_service=session_service)

    print("✅ Upgraded to persistent sessions!")
    print(f"   - Database: my_agent_data.db")
    print(f"   - Sessions will survive restarts!")
    
    # Test the runner
    response = await run_session(
        runner, 
        [ # These are the set of new queries, any previous conversation events will already be stored in SQLite database under the session name
            "Hi there, my name is Sarah! Can you tell me the capital of the New Zealand?", 
            "Hello! What is my name?",  # This time, the agent should remember!
        ],
        "test-db-session-01",
        session_service
    )
    
    # Testing Agents forgetfulness (It still remembers :D)
    response = await run_session(
        runner, 
        [
            "What is the Capital of India?",
            "Who am I?"
        ],
        "test-db-session-01",
        session_service
    )
    
    response = await run_session(
        runner, 
        [
            "What was my name again?",
            "Which countries I asked about earlier?"
        ],
        "test-db-session-02", # Changed session name, does not know this one
        session_service
    )
    

def check_data_in_db(db_name):
    with sqlite3.connect(f"Day3/sample-agent/{db_name}") as connection:
        cursor = connection.cursor()
        result = cursor.execute(
            "select app_name, session_id, author, content from events"
        )
        print([_[0] for _ in result.description])
        for each in result.fetchall():
            print(each)


# context compaction
async def PersistentAgentWithContextCompaction():
    
    chatbot_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot with persistent memory",
    )
    
    # Re-define our app with Events Compaction enabled
    research_app_compacting = App(
        name="research_app_compacting",
        root_agent=chatbot_agent,
        # This is the new part!
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=3,  # Trigger compaction every 3 invocations
            overlap_size=1,  # Keep 1 previous turn for context
        ),
    )

    db_url = "sqlite:///Day3/sample-agent/research_agent_data.db"  # Local SQLite file
    session_service = DatabaseSessionService(db_url=db_url)

    # Create a new runner for our upgraded app
    research_runner_compacting = Runner(
        app=research_app_compacting, session_service=session_service
    )
    
    print("✅ Research App upgraded with Events Compaction!")
    print(f"   - Database: my_agent_data.db")
    print(f"   - Sessions will survive restarts!")
    
    # # Turn 1
    # response = await run_session(
    #     research_runner_compacting,
    #     "What is the latest news about AI in healthcare?",
    #     "compaction_demo",
    #     session_service
    # )

    # # Turn 2
    # response = await run_session(
    #     research_runner_compacting,
    #     "Are there any new developments in drug discovery?",
    #     "compaction_demo",
    #     session_service
    # )

    # # Turn 3 - Compaction should trigger after this turn!
    # response = await run_session(
    #     research_runner_compacting,
    #     "Tell me more about the second development you found.",
    #     "compaction_demo",
    #     session_service
    # )

    # # Turn 4
    # response = await run_session(
    #     research_runner_compacting,
    #     "Who are the main companies involved in that?",
    #     "compaction_demo",
    #     session_service
    # )
    
    async def checkForCompactionEvent():
        # Get the final session state
        final_session = await session_service.get_session(
            app_name=research_runner_compacting.app_name,
            user_id=USER_ID,
            session_id="compaction_demo",
        )

        print("--- Searching for Compaction Summary Event ---")
        found_summary = False
        for event in final_session.events:
            # Compaction events have a 'compaction' attribute
            if event.actions and event.actions.compaction:
                print("\n✅ SUCCESS! Found the Compaction Event:")
                print(f"  Author: {event.author}")
                print(f"\n Compacted information: {event}")
                found_summary = True
                break

        if not found_summary:
            print(
                "\n❌ No compaction event found. Try increasing the number of turns in the demo."
            )

    await checkForCompactionEvent()

# Creating custom tools for Session state management (transferable characteristic across sessions, here username and user_country)
# Define scope levels for state keys (following best practices)
USER_NAME_SCOPE_LEVELS = ("temp", "user", "app")

# This demonstrates how tools can write to session state using tool_context.
# The 'user:' prefix indicates this is user-specific data.
def save_userinfo(tool_context: ToolContext, user_name: str, country: str) -> Dict[str, Any]:
    """
    Tool to record and save user name and country in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country
    """
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country

    return {"status": "success"}

# This demonstrates how tools can read from session state.
def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and country from session state.
    """
    # Read from session state
    user_name = tool_context.state.get("user:name", "Username not found")
    country = tool_context.state.get("user:country", "Country not found")

    return {"status": "success", "user_name": user_name, "country": country}
print("✅ Tools created.")    
    

async def AgentWithSessionStateTools():
    # Create an agent with session state tools
    root_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="""A text chatbot.
        Tools for managing user context:
        * To record username and country when provided use `save_userinfo` tool. 
        * To fetch username and country when required use `retrieve_userinfo` tool.
        """,
        tools=[save_userinfo, retrieve_userinfo],  # Provide the tools to the agent
    )

    # Set up session service and runner
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="default")

    print("✅ Agent with session state tools initialized!")

    # Test conversation demonstrating session state
    response = await run_session(
        runner,
        [
            "Hi there, how are you doing today? What is my name?",  # Agent shouldn't know the name yet
            "My name is Sam. I'm from Poland.",  # Provide name - agent should save it
            "What is my name? Which country am I from?",  # Agent should recall from session state
        ],
        "state-demo-session",
        session_service
    )
    
    async def checkSessionState(session_name: str):
        # Retrieve the session and inspect its state
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_name
        )

        print("\nCurrent Session: ",session.id)
        print("Session State Contents:")
        print(session.state)
        print("\n🔍 Notice the 'user:name' and 'user:country' keys storing our data!")

    await checkSessionState("state-demo-session")
    
    
    # Start a completely new session - the agent won't know our name
    response = await run_session(
        runner,
        ["Hi there, how are you doing today? What is my name?"],
        "new-isolated-session",
        session_service
    ) # Expected: The agent won't know the name because this is a different session
    
    await checkSessionState("new-isolated-session")
    
    
if __name__ == "__main__":
    # asyncio.run(InMemoryAgent())
    # asyncio.run(PersistentAgent())
    # check_data_in_db("my_agent_data.db")  
    # check_data_in_db("research_agent_data.db")  
    # asyncio.run(PersistentAgentWithContextCompaction())
    asyncio.run(AgentWithSessionStateTools())