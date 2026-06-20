import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_insights(simulation_results: dict) -> str:
    """
    Takes the JSON output from the SimPy simulation and asks GPT-5.4 
    via Azure AI Foundry to analyze it for bottlenecks and recommendations.
    """
    # Grab Azure AI Foundry credentials from the environment
    # Note: For the v1 API, the endpoint should end in /openai/v1
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4") 

    if not endpoint or not api_key:
        raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables must be set.")

    # Using the standard OpenAI client but pointing it to Azure's v1 endpoint
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
        default_headers={"api-key": api_key} # Azure uses this specific header for keys
    )

    prompt = f"""
    You are an expert business process analyst. 
    I have run a Monte Carlo simulation on an event log. Here are the results:

    {json.dumps(simulation_results, indent=2)}

    Review the data and provide:
    1. The primary bottleneck (look closely at resource utilization near 100%).
    2. The impact of this bottleneck on the overall cycle time.
    3. 2-3 actionable, plain-English recommendations to improve the process.

    Keep the tone professional, concise, and executive-friendly. Do not just repeat the raw JSON numbers back to me; interpret what they mean for the business.
    """

    response = client.chat.completions.create(
        model=deployment_name, 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content