import asyncio
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
print("✅ ADK components imported successfully.")

# agent automatically retries if something goes wrong
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

async def ResearchSummarize():
        
    """Example: Research and Summarization system."""
    # Uses a root_agent and makes a sequential call to agents as mentioned in its instructions. (not reliable since the LLM decides to when use the tool)

    research_agent = Agent(
        name="ResearchAgent",
        model=Gemini(
            model='gemini-2.5-flash',
            retry_options=retry_config
        ),
        instruction="""You are a specialized research agent. Your only job is to use the google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
        tools=[google_search],
        output_key="research_findings",  # The result of this agent will be stored in the session state with this key.
    )
    print("✅ research_agent created.")

    summarizer_agent = Agent(
        name="SummarizerAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        # The instruction is modified to request a bulleted list for a clear output format, given the research_findings from the ResearchAgent
        instruction="""Read the provided research findings: \n{research_findings} \n\nCreate a concise summary as a bulleted list with 3-5 key points.""",
        output_key="final_summary",
    )
    print("✅ summarizer_agent created.")
    
    root_agent = Agent(
        name="ResearchCoordinator",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
            instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow. Execute the workflow in the following steps:
                    1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
                    2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
                    3. Finally, present the final summary clearly to the user as your response.""",

        # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
        tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
    )
    print("✅ ResearchCordinator_agent created.")
    
    runner = InMemoryRunner(agent=root_agent)
    print("✅ Runner created.")
    
    response = await runner.run_debug(
        "What are the latest advancements in quantum computing and what do they mean for AI?"
    )    
    

async def BlogPostCreation():
    
    """Let's build a system with three specialized agents:
        Outline Agent - Creates a blog outline for a given topic
        Writer Agent - Writes a blog post
        Editor Agent - Edits a blog post draft for clarity and structure"""
            
    # Uses SequentialAgent, every agent is called in a predefined sequence of tasks. No LLM decides the executions here.
    
    # Outline Agent: Creates the initial blog post outline.
    outline_agent = Agent(
        name="OutlineAgent",
        model=Gemini(
            model='gemini-2.5-flash',
            retry_options=retry_config
        ),
        instruction="""Create a blog outline for the given topic with:
                        1. A catchy headline
                        2. An introduction hook
                        3. 3-5 main sections with 2-3 bullet points for each
                        4. A concluding thought""",
        output_key="blog_outline",  # The result of this agent will be stored in the session state with this key.
    )
    print("✅ outline_agent created.")
    
    # Writer Agent: Writes the full blog post based on the outline from the previous agent.
    writer_agent = Agent(
        name="WriterAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        # The `{blog_outline}` placeholder automatically injects the state value from the previous agent's output.
        instruction="""Following this outline strictly: {blog_outline}
        Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
        output_key="blog_draft",  # The result of this agent will be stored with this key.
    )
    print("✅ writer_agent created.")
    
    # Editor Agent: Edits and polishes the draft from the writer agent.
    editor_agent = Agent(
        name="EditorAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        # This agent receives the `{blog_draft}` from the writer agent's output.
        instruction="""Edit this draft: {blog_draft}
        Your task is to polish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity.""",
        output_key="final_blog",  # This is the final output of the entire pipeline.
    )
    print("✅ editor_agent created.")
    
    # All the agents will have to execute sequentially after one and another
    root_agent = SequentialAgent(
        name="BlogPipeline",
        sub_agents=[outline_agent, writer_agent, editor_agent],
    )
    print("✅ Sequential Agent created.")
    
    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug(
        "Write a blog post about the benefits of multi-agent systems for software developers"
    )


async def ParallelMultiTopicResearch():
    """Let's build a system with four agents:
        Tech Researcher - Researches AI/ML news and trends
        Health Researcher - Researches recent medical news and trends
        Finance Researcher - Researches finance and fintech news and trends
        Aggregator Agent - Combines all research findings into a single summary""" 
        
    # All agents work in parallel, no one has to wait for another to finish their task
    
    # Tech Researcher: Focuses on AI and ML trends.
    tech_researcher = Agent(
        name="TechResearcher",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Research the latest AI/ML trends. Include 3 key developments, the main companies involved, and the potential impact. Keep the report very concise (100 words).""",
        tools=[google_search],
        output_key="tech_research",  # The result of this agent will be stored in the session state with this key.
    )
    print("✅ tech_researcher created.")
    
    # Health Researcher: Focuses on medical breakthroughs.
    health_researcher = Agent(
        name="HealthResearcher",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Research recent medical breakthroughs. Include 3 significant advances, their practical applications, and estimated timelines. Keep the report concise (100 words).""",
        tools=[google_search],
        output_key="health_research",  # The result will be stored with this key.
    )
    print("✅ health_researcher created.")

    # Finance Researcher: Focuses on fintech trends.
    finance_researcher = Agent(
        name="FinanceResearcher",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Research current fintech trends. Include 3 key trends, their market implications, and the future outlook. Keep the report concise (100 words).""",
        tools=[google_search],
        output_key="finance_research",  # The result will be stored with this key.
    )
    print("✅ finance_researcher created.")
    
    # The AggregatorAgent runs *after* the parallel step to synthesize the results.
    aggregator_agent = Agent(
        name="AggregatorAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
        instruction="""Combine these three research findings into a single executive summary:

        **Technology Trends:**
        {tech_research}
        
        **Health Breakthroughs:**
        {health_research}
        
        **Finance Innovations:**
        {finance_research}
        
        Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",
        output_key="executive_summary",  # This will be the final output of the entire system.
    )
    print("✅ aggregator_agent created.")
    
    # Define Parallel Workflow
    parallel_research_team = ParallelAgent(
        name="ParallelResearchTeam",
        sub_agents=[tech_researcher, health_researcher, finance_researcher]
    )
    print("✅ parallel_research_team created.")
    
    # This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
    root_agent = SequentialAgent(
        name="ResearchSystem",
        sub_agents=[parallel_research_team, aggregator_agent],
    )
    print("✅ Parallel and Sequential Agents created.")
        
    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug(
        "Run the daily executive briefing on Tech, Health, and Finance"
    )
   
   
async def IterativeStoryRefinement():
    """Let's build a system with two agents:
        Writer Agent - Writes a draft of a short story
        Critic Agent - Reviews and critiques the short story to suggest improvements"""
    
    # This agent runs ONCE at the beginning to create the first draft.
    initial_writer_agent = Agent(
        name="InitialWriterAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""Based on the user's prompt, write the first draft of a short story (around 100-150 words).
        Output only the story text, with no introduction or explanation.""",
        output_key="current_story",  # Stores the first draft in the state.
    )
    print("✅ initial_writer_agent created.")
    
    # This agent's only job is to provide feedback or the approval signal. It has no tools.
    critic_agent = Agent(
        name="CriticAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""You are a constructive story critic. Review the story provided below.
        Story: {current_story}
        
        Evaluate the story's plot, characters, and pacing.
        - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
        - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
        output_key="critique",  # Stores the feedback in the state.
    )
    print("✅ critic_agent created.")
    
    # This is the function that the RefinerAgent will call to exit the loop.
    def exit_loop():
        """Called ONLY when the critique is 'APPROVED'. Returning a termination sentinel so an outer loop (LoopAgent) can decide to stop iterating if it inspects tool output.

        If the underlying LoopAgent implementation does not automatically stop, you can instead replace the LoopAgent usage with a manual loop that checks for this flag.
        """
        return {
            "status": "approved",
            "message": "Story approved. Exiting refinement loop.",
            "terminate_loop": True  # Sentinel flag for early stopping
        }
    print("✅ exit_loop function created.")
    
    # This agent refines the story based on critique OR calls the exit_loop function.
    refiner_agent = Agent(
        name="RefinerAgent",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        instruction="""You are a story refiner. You have a story draft and critique.

        Story Draft: {current_story}
        Critique: {critique}

        Your task:
        - If the critique is EXACTLY "APPROVED" and not previously terminated, CALL the `exit_loop` function and do NOTHING else.
        - Otherwise, rewrite the story draft incorporating the feedback.

        IMPORTANT: Never modify an already approved story. Only call the tool once. If state already contains a prior approval (e.g., terminate_loop=True), do not call the tool again and simply return the existing story unchanged.""",
        output_key="current_story",  # It overwrites the story with the new, refined version.
        tools=[FunctionTool(exit_loop)]  # The tool is now correctly initialized with the function reference.
    )
    print("✅ refiner_agent created.")
    
    # The LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
    story_refinement_loop = LoopAgent(
        name="StoryRefinementLoop",
        sub_agents=[critic_agent, refiner_agent],
        max_iterations=2,  # Prevents infinite loops
    )
    
    # The root agent is a SequentialAgent that defines the overall workflow: Initial Write -> Refinement Loop.
    root_agent = SequentialAgent(
        name="StoryPipeline",
        sub_agents=[initial_writer_agent, story_refinement_loop],
    )
    print("✅ Loop and Sequential Agents created.")
    
    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug(
        "Write a short horror story about a lighthouse keeper who discovers a mysterious, glowing map"
    )
        

if __name__ == "__main__":
    # asyncio.run(ResearchSummarize())
    # asyncio.run(BlogPostCreation())
    # asyncio.run(ParallelMultiTopicResearch())
    asyncio.run(IterativeStoryRefinement())