import asyncio
from json import load
from unittest import runner
from dotenv import load_dotenv
from elevenlabs import Llm
from flask import session
from torch import mode
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService, VertexAiMemoryBankService
from google.adk.tools import load_memory, preload_memory
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

# from adk_imports import *
print("✅ ADK components imported successfully.")

# Types of MemoryService -> InMemoryMemoryService (Stores raw events, Keyword matching retrieval), VertexAiMemoryBankService (Intelligently consolidates before storing i.e. store summary of events, Retrieval Semantic search via embeddings)

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

APP_NAME = "MemoryDemoApp"  # Application
USER_ID = "demo_user"  # User
SESSION = "default"  # Session


async def run_session(runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default", session_service: InMemorySessionService = InMemorySessionService):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    # Create or retrieve session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Convert single query to list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        # Stream agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Agent > {text}")
print("✅ Helper functions defined.")

async def MemorySavingAgent():
    """Agent that only saves current session events to the memory. (cannot access memory)"""
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    user_agent = LlmAgent(
        name="MemoryDemoAgent",
        model=Gemini(
            model='gemini-2.5-flash-lite',
            retry_options=retry_config
        ),
        instruction="Answer user questions in simple words."
    )
    print("✅ Agent Created!")

    runner = Runner(
        agent=user_agent,
        app_name=APP_NAME, 
        session_service=session_service,
        memory_service=memory_service
    )
    print("✅ Agent and Runner created with memory support!")

    # User tells agent about their favorite color
    response = await run_session(
        runner,
        "My favorite color is blue-green. Can you write a Haiku about it?",
        "conversation-01",  # Session ID
        session_service
    )
    # The runner will store the raw events in the session.id
    
    # For the events to be accesed by all the sessions using the memory, Call add_session_to_memory() and pass the session object
    
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id='conversation-01')
    
    print("📝 Session contains:")
    for event in session.events:
        text = event.content.parts[0].text if event.content and event.content.parts else "(empty)"
        print(f"  {event.content.role}: {text}...")

    # This is the key method! (manual)
    await memory_service.add_session_to_memory(session=session) 
    print("✅ Session added to memory!")


async def MemorySearchingAgent():
    """Agent that can access previously stored event in memory."""
    
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    user_agent = LlmAgent(
        name="MemoryDemoAgent",
        model=Gemini(
            model='gemini-2.5-flash-lite',
            retry_options=retry_config
        ),
        instruction="Answer user questions in simple words.",
        tools=[load_memory] # preload_memory
    )
    
    # ADK provides two built-in tools for memory retrieval:
    # load_memory (Reactive)
        # Agent decides when to search memory
        # Only retrieves when the agent thinks it's needed
        # More efficient (saves tokens)
        # Risk: Agent might forget to search
        
    # preload_memory (Proactive)
        # Automatically searches before every turn
        # Memory always available to the agent
        # Guaranteed context, but less efficient
        # Searches even when not needed
    
    print("✅ Agent Created!")

    runner = Runner(
        agent=user_agent,
        app_name=APP_NAME, 
        session_service=session_service,
        memory_service=memory_service
    )
    print("✅ Agent and Runner created with memory support!")

    # User tells agent about their favorite color
    response = await run_session(
        runner,
        "My birthday is on March 15th.", 
        "birthday-session-01",  # Session ID
        session_service
    )
    # The runner will store the raw events in the session.id
    
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id="birthday-session-01")
    
    print("📝 Session contains:")
    for event in session.events:
        text = event.content.parts[0].text if event.content and event.content.parts else "(empty)"
        print(f"  {event.content.role}: {text}...")
    
    # For the events to be accesed by all the sessions using the memory, Call add_session_to_memory() and pass the session object
    
    await memory_service.add_session_to_memory(session=session) # This is the key method! (manual)
    print("✅ Birthday session saved to memory!")
    
    # Test if the memory got updated
    response = await run_session(
        runner,
        "When is my birthday?", 
        "birthday-session-02",  # Different Session ID
        session_service
    )
    
    async def SearchMemoryManually(memory_service):
        # Search for color preferences
        search_response = await memory_service.search_memory(
            app_name=APP_NAME, user_id=USER_ID, query="What is the user's favorite color?"
        )

        print("\n\n🔍 Search Results:")
        print(f"  Found {len(search_response.memories)} relevant memories")
        print()

        for memory in search_response.memories:
            if memory.content and memory.content.parts:
                text = memory.content.parts[0].text[:80]
                print(f"  [{memory.author}]: {text}...")
    
    await SearchMemoryManually(memory_service)
    
    
async def AgentWithAutomaticMemoryStorageandSearching():
    """Agent stores session events to memeory automatically and search the memory everytime using preload_memory tool."""
    
    async def auto_save_to_memory(callback_context: CallbackContext): # This is a custom tool agent will call whenever it wishes to save memory.
        """Automatically save session to memory after each agent turn."""
        await callback_context._invocation_context.memory_service.add_session_to_memory(
            callback_context._invocation_context.session
        )
    print("✅ Callback created.")
    
    user_agent = LlmAgent(
        name="AutoMemoryAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="Answer user questions",
        tools=[preload_memory],
        after_agent_callback=auto_save_to_memory  # Saves after each turn!
    )
    
    # before_agent_callback → Runs before agent starts processing a request
    # after_agent_callback → Runs after agent completes its turn
    # before_tool_callback / after_tool_callback → Around tool invocations
    # before_model_callback / after_model_callback → Around LLM calls
    # on_model_error_callback → When errors occur
        
    
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()
    
    runner = Runner(
        agent=user_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service
    )
    
    # Test 1: Tell the agent about a gift (first conversation)
    # The callback will automatically save this to memory when the turn completes
    response = await run_session(
        runner, 
        "I gifted a new toy car to my nephew on his 1st birthday!",
        "auto-save-test-01",
        session_service
    )
    
    
    # Test 2: Ask about the gift in a NEW session (second conversation)
    # The agent should retrieve the memory using preload_memory and answer correctly
    response = await run_session(
        runner, 
        "What did I gift my nephew?",
        "auto-save-test-02",
        session_service
    )
    

    
if __name__ == "__main__":
    # asyncio.run(MemorySavingAgent())
    # asyncio.run(MemorySearchingAgent())
    asyncio.run(AgentWithAutomaticMemoryStorageandSearching())