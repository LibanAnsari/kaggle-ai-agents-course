import os
import random
import time
import vertexai
# from kaggle_secrets import UserSecretsClient # For kaggle only
from vertexai import agent_engines
print("✅ Imports completed successfully")

# Set up Cloud Credentials in Kaggle
# user_secrets = UserSecretsClient()
# user_credential = user_secrets.get_gcloud_credential()
# user_secrets.set_tensorflow_credential(user_credential)
print("✅ Cloud credentials configured")

## Set your PROJECT_ID
PROJECT_ID = "kaggle-day5-31"  # TODO: Replace with your project ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

if PROJECT_ID == "your-project-id" or not PROJECT_ID:
    raise ValueError("⚠️ Please replace 'your-project-id' with your actual Google Cloud Project ID.")
print(f"✅ Project ID set to: {PROJECT_ID}")

regions_list = ["europe-west1", "europe-west4", "us-east4", "us-west1"]
deployed_region = random.choice(regions_list)
print(f"✅ Selected deployment region: {deployed_region}")

# !adk deploy agent_engine --project=$PROJECT_ID --region=$deployed_region sample_agent --agent_engine_config_file=sample_agent/.agent_engine_config.json
# The adk deploy agent_engine command:
# Packages your agent code (sample_agent/ directory)
# Uploads it to Agent Engine
# Creates a containerized deployment
# Outputs a resource name like: projects/PROJECT_NUMBER/locations/REGION/reasoningEngines/ID


######## TEST ########

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=deployed_region)

# Get the most recently deployed agent
agents_list = list(agent_engines.list())
if agents_list:
    remote_agent = agents_list[0]  # Get the first (most recent) agent
    client = agent_engines
    print(f"✅ Connected to deployed agent: {remote_agent.resource_name}")
else:
    print("❌ No agents found. Please deploy first.")
# This cell retrieves your deployed agent:
# Initializes the Vertex AI SDK with your project and region
# Lists all deployed agents in that region
# Gets the first one (most recently deployed)
# Stores it as remote_agent for testing    

async def test():
    async for item in remote_agent.async_stream_query(
        message="What is the weather in Tokyo?",
        user_id="user_42",
    ):
        print(item)
    """You'll see multiple items printed:
        Function call - Agent decides to call get_weather tool
        Function response - Result from the tool (weather data)
        Final response - Agent's natural language answer"""
    
agent_engines.delete(resource_name=remote_agent.resource_name, force=True)
print("✅ Agent successfully deleted")