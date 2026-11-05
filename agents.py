
import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, LLM




# Load environment variables
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR / ".env",
    override=True
)


# Check Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )


# Configure Groq LLM
# Configure Groq LLM
llm = LLM(
    model=os.getenv(
        "GROQ_MODEL",
        "groq/openai/gpt-oss-20b"
    ),
    api_key=GROQ_API_KEY,
    temperature=0.2
)


# ------------------------------------------------
# AGENT 1: DISCOVERY AGENT
# ------------------------------------------------

discovery_agent = Agent(

    role="Academic Research Paper Discovery Specialist",

    goal=(
        "Identify genuine and relevant academic research papers "
        "for the given research topic. Preserve original titles, "
        "authors, publication years, DOI and source URLs."
    ),

    backstory=(
        "You are an academic research discovery specialist. "
        "You carefully examine scholarly publication metadata. "
        "You never invent paper titles, authors, DOI numbers "
        "or publication details."
    ),

    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ------------------------------------------------
# AGENT 2: SUMMARIZATION AGENT
# ------------------------------------------------

summarization_agent = Agent(

    role="Academic Research Paper Summarization Expert",

    goal=(
        "Prepare accurate and structured summaries of research "
        "papers using only their supplied abstracts or full texts."
    ),

    backstory=(
        "You are an academic research analyst skilled in "
        "understanding research objectives, methodologies, "
        "findings and limitations. You clearly identify "
        "information that is unavailable in the supplied text."
    ),

    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ------------------------------------------------
# AGENT 3: LITERATURE REVIEW AGENT
# ------------------------------------------------

literature_review_agent = Agent(

    role="Academic Literature Review Specialist",

    goal=(
        "Compare research papers, identify common themes, "
        "compare methodologies, findings and limitations, "
        "and prepare a structured literature review."
    ),

    backstory=(
        "You are an experienced academic researcher specializing "
        "in systematic literature analysis. You compare papers "
        "critically and support your observations with the "
        "provided paper references."
    ),

    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ------------------------------------------------
# AGENT 4: RESEARCH GAP AGENT
# ------------------------------------------------

research_gap_agent = Agent(

    role="Research Gap Identification Specialist",

    goal=(
        "Identify potential research gaps based on the "
        "limitations, findings and underexplored areas "
        "in the supplied research papers."
    ),

    backstory=(
        "You are an academic research strategist. You examine "
        "limitations and compare existing research to identify "
        "potential opportunities for future investigation. "
        "You distinguish evidence-supported observations "
        "from hypotheses requiring further validation."
    ),

    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ------------------------------------------------
# AGENT 5: REPORT GENERATION AGENT
# ------------------------------------------------

report_agent = Agent(

    role="Academic Research Report Writing Specialist",

    goal=(
        "Prepare a comprehensive research assistance report "
        "containing paper details, summaries, literature review, "
        "potential research gaps, future directions and references."
    ),

    backstory=(
        "You are an academic technical writer experienced in "
        "preparing research reports and literature surveys. "
        "You preserve source references and never fabricate "
        "citations, quotations or research findings."
    ),

    llm=llm,
    verbose=True,
    allow_delegation=False
)


# ------------------------------------------------
# DISPLAY AGENT INFORMATION
# ------------------------------------------------

if __name__ == "__main__":

    agents = [
        discovery_agent,
        summarization_agent,
        literature_review_agent,
        research_gap_agent,
        report_agent
    ]

    print("\nCREWAI RESEARCH ASSISTANT")
    print("=" * 50)

    for number, agent in enumerate(agents, start=1):

        print(f"\nAgent {number}: {agent.role}")
        print("Goal:", agent.goal)

    print("\nTotal agents created:", len(agents))
    print("Groq LLM configuration loaded successfully.")
