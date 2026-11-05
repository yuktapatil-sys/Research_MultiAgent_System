
# Fix CrewAI cache_breakpoint issue with Groq
import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda msg: msg
from crewai import Crew, Process

from agents import (
    discovery_agent,
    summarization_agent,
    literature_review_agent,
    research_gap_agent,
    report_agent
)

from tasks import (
    discovery_task,
    summarization_task,
    literature_review_task,
    research_gap_task,
    report_task
)


# --------------------------------------------------
# CREATE THE RESEARCH CREW
# --------------------------------------------------

research_crew = Crew(

    agents=[
        discovery_agent,
        summarization_agent,
        literature_review_agent,
        research_gap_agent,
        report_agent
    ],

    tasks=[
        discovery_task,
        summarization_task,
        literature_review_task,
        research_gap_task,
        report_task
    ],

    process=Process.sequential,

    verbose=True
)


# --------------------------------------------------
# RUN THE CREW
# --------------------------------------------------

def run_research(topic, papers_data):

    print("\nStarting Multi-Agent Research Assistant...")
    print("Research Topic:", topic)

    result = research_crew.kickoff(
        inputs={
            "research_topic": topic,
            "papers_data": papers_data
        }
    )

    return result


# --------------------------------------------------
# BASIC CONFIGURATION TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("\nCREWAI RESEARCH CREW")
    print("=" * 45)

    print("Agents connected:", len(research_crew.agents))
    print("Tasks connected:", len(research_crew.tasks))
    print("Process: Sequential")

    print("\nCrew configuration completed successfully.")
