#
# This is a supply chain risk analysis application built using CrewAI.
#
# What is CrewAI?
# CrewAI is an open-source framework that enables developers to build intelligent,
# multi-agent systems. It allows you to define a team of AI agents, give them
# specific roles and goals, and then have them work together collaboratively
# to accomplish complex tasks. It's built on top of LangChain and is designed
# for orchestrating a series of tasks performed by different agents.
#
# What This Code Does:
# This script creates a crew of two AI agents: a "Supply Chain Risk Analyst" and a
# "Supply Chain Strategist." The agents are given distinct roles, goals, and backstories
# to guide their behavior. The script then defines a sequential workflow of three tasks:
# 1. Research a given supply chain topic to identify potential risks.
# 2. Analyze the identified risks to assess their probability and severity.
# 3. Develop concrete mitigation strategies for the most significant risks.
#
# The agents work together automatically to complete these tasks, with the output of
# each step feeding into the next. The final result is a comprehensive report on
# supply chain risks and solutions for a user-provided topic.
#

import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI

# Set up the environment for CrewAI and LangChain
os.environ["OPENAI_API_KEY"] = "sk-..." # This key is not used in the provided model.
os.environ["SERPER_API_KEY"] = "Your_SERPER_API_KEY" # Placeholder. Please replace with your actual API key.
os.environ["DUCKDUCKGO_API_KEY"] = "" # The key is not required for DuckDuckGoSearchRun

# Initialize the search tool
duckduckgo_search_tool = DuckDuckGoSearchRun()

# Define the LLM model to be used by the agents
# Use gemini-2.5-flash-preview-05-20 as the model
llm = GoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")

# --- Define the Agents ---
# Each agent has a specific role, goal, and backstory.
# This helps guide the AI's behavior and thought process.

# The Risk Analyst Agent
# This agent's primary job is to research and identify risks.
risk_analyst = Agent(
    role="Supply Chain Risk Analyst",
    goal="""
        Identify and analyze potential risks in the supply chain for a specific product.
        Focus on geopolitical, logistical, and environmental factors.
    """,
    backstory="""
        A seasoned supply chain expert with a keen eye for potential disruptions.
        You are known for your meticulous research and ability to foresee problems
        before they happen. Your work is crucial for business resilience.
    """,
    verbose=True, # Set to True to see the agent's thought process
    allow_delegation=False, # This agent will not delegate tasks
    llm=llm,
    tools=[duckduckgo_search_tool]
)

# The Strategist Agent
# This agent's primary job is to create mitigation strategies.
strategist = Agent(
    role="Supply Chain Strategist",
    goal="""
        Develop actionable mitigation strategies based on the risks identified by the
        Risk Analyst. The strategies should focus on building resilience and
        reducing potential negative impacts.
    """,
    backstory="""
        A creative and practical problem-solver who excels at turning complex
        problems into simple, effective solutions. You specialize in building
        robust and agile supply chains that can withstand any disruption.
    """,
    verbose=True,
    llm=llm,
    allow_delegation=False,
)

# --- Define the Tasks ---
# Each task has a description and is assigned to a specific agent.

# Task 1: Research and identify risks for a given topic
research_task = Task(
    description="""
        Conduct comprehensive research on the supply chain of {topic}.
        Identify all potential risks related to political instability,
        logistical bottlenecks, and environmental regulations.
        Provide a detailed report of your findings.
    """,
    expected_output="""
        A bulleted list of potential risks, including a brief description
        of each and its potential impact on the supply chain.
    """,
    agent=risk_analyst
)

# Task 2: Analyze the identified risks
analysis_task = Task(
    description="""
        Analyze the risks identified in the research report. For each risk,
        assess its probability and severity (e.g., high, medium, low).
        Explain the reasoning behind your assessment.
    """,
    expected_output="""
        A structured analysis report where each risk is assessed for its
        probability and severity, with a clear explanation for the rating.
    """,
    agent=risk_analyst
)

# Task 3: Develop mitigation strategies
mitigation_task = Task(
    description="""
        Based on the risk analysis, develop at least three concrete
        mitigation strategies. The strategies should be practical and
        aimed at making the supply chain more resilient.
    """,
    expected_output="""
        A clear and concise list of three or more actionable strategies,
        with a brief explanation of how each strategy addresses the risks.
    """,
    agent=strategist
)

# --- Create the Crew ---
# The Crew orchestrates the agents and tasks.

# The Crew will process the tasks in a sequential order.
# The output of one task becomes the input for the next.
crew = Crew(
    agents=[risk_analyst, strategist],
    tasks=[research_task, analysis_task, mitigation_task],
    verbose=2, # Set to 2 to see the detailed thought process of the agents
    process=Process.sequential
)

# --- Kickoff the process ---
# The kickoff method starts the entire workflow.
if __name__ == "__main__":
    topic = input("Enter a supply chain topic to analyze (e.g., 'lithium-ion batteries'): ")
    result = crew.kickoff(inputs={"topic": topic})
    print("\n\n##################################")
    print("## Final Supply Chain Report:")
    print("##################################\n")
    print(result)
