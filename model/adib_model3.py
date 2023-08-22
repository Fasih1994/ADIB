import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

from langchain.agents.agent_toolkits.pandas.base import _get_functions_prompt_and_tools
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import get_openai_callback




import pandas as pd


load_dotenv('../.flaskenv')
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


prefix = ("You are the CFO of the bank and you're analyzing the ADIB balance sheet. Always return greetings as `I am great` etc."
          "Here are some insights from the data:"
          "- You can use the Pandas DataFrame named 'adib' to analyze financial data which is already imported."
          "- Use appropriate filters to focus on specific information."
          "- Columns are:'Chart of Accounts','2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025','2026', '2027', '2028'"
          """- 'Chart of Accounts' column contain these values: 
'Gross revenue from funds', 'Distribution to depositors',
'Funded income', 'Investment income', 'Fees and commission income',
'Foreign exchange income', 'Other income', 'Non-funded income',
'Revenues', 'Net income before zakat and tax', 'Zakat and tax',
'Net income after zakat and tax', 'Non-controlling interest',
'Net income attributable to equity holders of the bank',
'Cash and balances with central banks',
'Due from financial institutions', 'Customer financing', 'Investments',
'Investment in associates', 'Investment and development properties',
'Other assets/ Property and Equipement/ Goodwill and Intangibles',
'Total assets', 'Due to financial institutions', 'Total deposits',
'Other liabilities', 'Total liabilities', 'Share capital ',
'Legal reserve', 'General reserve', 'Credit risk reserve',
'Retained earnings', 'Proposed dividends', 'Other reserves',
'Equity attributable to shareholders of the bank', 'Tier 1 sukuk ',
'Non - controlling interest',
'Equity attributable to equity holders of the bank',
'Total liabilities and equity', 'Customer financing gross',
'Non-performing financing (NPA)', 'NPA ratio', 'NPA coverage ratio',
'NPA coverage ratio with collaterals', 'RWA', 'Operating Income',
'Total Revenue', 'Tier 1 Capital Ratio', 'Capital Adequacy Ratio',
'Common Equity Tier 1 Ratio ', 'Depreciation    (Cost)',
'Amortisation of intangibles  (Cost)',
'Card related fees and commission expense (Cost)',
'Other fees and commission expenses (Cost)',
'Salaries and wages (Cost)', 'End of service benefits  (Cost)',
'Other staff expenses (Cost)', 'Legal and professional expenses (Cost)',
'Premises expenses (Cost)', 'Marketing and advertising expenses (Cost)',
'Communication expenses  (Cost)', 'Technology related expenses  (Cost)',
'Finance cost on lease liabilities  (Cost)',
'Other operating expenses  (Cost)',
'Murabaha and other Islamic financing (Cost)', 'Ijara financing (Cost)',
'Direct write-off net of recoveries (Cost)',
'Investment in sukuk measured at amortised cost (Cost)',
'Other Impairment charges (Cost)', 'Cost-to-Income Ratio (Cost)',
'Efficiency Ratio (Cost)', 'Operating Profit Margin (Cost)'"""
    "- current year is 2023 and the numbers are in AED Millions"
    "- for all cost related queries check against all `Chart of account` values ending with '(Cost)'"
    "Following are the formulae for special cost heads:"
    "Operating expenses =  sum(all_cost_heads) - sum('Card related fees and commission expense (Cost)','Other fees and commission expenses (Cost)','Cost-to-Income Ratio (Cost)','Efficiency Ratio (Cost)', 'Operating Profit Margin (Cost)')"
    "Provision for impairment net = Murabaha and other Islamic financing (Cost)+ Ijara financing (Cost) +Direct write-off, net of recoveries (Cost)+Investment in sukuk measured at amortised cost (Cost)+Other Impairment charges (Cost)"
    "General and administrative expenses = Legal and professional expenses (Cost)+ Premises expenses (Cost) + Marketing and advertising expenses (Cost)+ Communication expenses  (Cost)+ Technology related expenses  (Cost)+Finance cost on lease liabilities  (Cost)+Other operating expenses"
    "Employees costs = Salaries and wages (Cost)+ End of service benefits  (Cost)+ Other staff expenses (Cost)"
    "For `Series.str.contains('some_value', regex=True)` Use regex = True first if it gives nan then use regex=False"
    "If asked for special cost heads ALWAYS compute then answer."
)

system_message = SystemMessage(content=prefix)

MEMORY_KEY = "chat_history" 
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
)

path = os.path.join(os.getcwd(),'data/ADIB_Financial data_With cost heads_V2.csv')
adib = pd.read_csv(path)
adib = adib.iloc[:,:13]
adib.index = adib['Chart of Accounts']

def handle_parsing_errors(error):
    msg = """Handel Parsing error yourself!"""
    return msg

def get_agent(chat_history:list = None):
    df_tool = PythonAstREPLTool(
        locals={"adib":adib}, 
        description=("A Python shell. Use this to execute python commands."
                     "Input should be a valid python command. When using this tool, sometimes output is abbreviated"
                     "- make sure it does not look abbreviated before using it in your answer."
                    )
    )

    llm = ChatOpenAI(temperature=0, model=os.environ['MODEL'])

    agent = OpenAIFunctionsAgent(
                llm=llm,
                prompt=prompt,
                tools=[df_tool]
    )

    memory = ConversationBufferWindowMemory(memory_key=MEMORY_KEY, return_messages=True, k=4)
    if len(chat_history)>1:
        for user_message, ai_message in chat_history[:-1]:
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(ai_message)
    # print(memory)
    agent = AgentExecutor(
        agent=agent,
        tools=[df_tool],
        memory=memory,
        max_iterations=7,
        verbose=True,
        handle_parsing_errors=handle_parsing_errors
    )

    return agent


def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    with get_openai_callback() as cb:
        response = agent.run(query)
        total_tokens = cb.total_tokens
        # prompt_token = cb.prompt_tokens
        # completion_token = cb.completion_tokens
        total_cost = cb.total_cost
        print(f"Tokens {total_tokens} costs {total_cost}")
    # Return the response converted to a string.
    return str(response)