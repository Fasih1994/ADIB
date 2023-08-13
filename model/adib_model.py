import os

from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import get_openai_callback
import pandas as pd

from propmts import ADIB_PROMPT

BASE_DIR = os.getcwd()

def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """
    # Prepare the prompt with query guidelines and formatting
    propmt = (ADIB_PROMPT + query)
    # Run the prompt through the agent and capture the response.
    with get_openai_callback() as cb:
        response = agent.run(propmt)
        # total_tokens = cb.total_tokens
        # prompt_token = cb.prompt_tokens
        # completion_token = cb.completion_tokens
        # total_cost = cb.total_cost
    # Return the response converted to a string.
    return str(response)

def handle_parsing_errors(error):
    msg = """Handel Parsing error yourself!"""
    return msg


def get_agent():
    dfs = []
    for file in os.listdir('data'):
        if file.endswith('.csv'):
            file_path = os.path.join(BASE_DIR, 'data', file)
            name = file.split('.')[0]
            df = pd.read_csv(file_path)
            df['name'] = name
            dfs.append(df)
    # final df
    df = pd.concat(dfs)
    
    # Create Agent
    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0.6), 
        df=df, 
        verbose=True,
        max_iterations=6,
        prompt = ADIB_PROMPT,
        handle_parsing_errors=handle_parsing_errors
    )
    return agent


