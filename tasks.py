
from crewai import Task

from agents import (
    discovery_agent,
    summarization_agent,
    literature_review_agent,
    research_gap_agent,
    report_agent
)


# --------------------------------------------------
# TASK 1: PAPER DISCOVERY
# --------------------------------------------------

discovery_task = Task(
    description="""
    Organize and examine the research papers retrieved from OpenAlex.

    Research Topic: {research_topic}

    Actual paper records retrieved from OpenAlex:
    {papers_data}

    For every paper, identify:
    - Title
    - Authors
    - Publication year
    - Abstract
    - DOI
    - Journal or publication venue
    - Source URL

    Use only the supplied paper records.
    Do not invent papers, authors, DOIs or publication details.
    Clearly identify records with missing information.
    """,

    expected_output="""
    A structured list of the retrieved research papers with
    their verified metadata and abstracts.
    """,

    agent=discovery_agent
)


# --------------------------------------------------
# TASK 2: PAPER SUMMARIZATION
# --------------------------------------------------

summarization_task = Task(
    description="""
    Summarize every research paper identified in the
    previous discovery task.

    For each paper, explain:
    1. Research problem
    2. Research objectives
    3. Methodology or techniques used
    4. Important findings
    5. Limitations, if mentioned in the abstract or paper

    Use only the information available in the paper records.
    Do not assume that an abstract contains full paper details.
    Mark unavailable information as "Not specified".
    """,

    expected_output="""
    Individual research summaries for all discovered papers,
    organized by title, with the problem, objectives,
    methodology, findings and limitations.
    """,

    agent=summarization_agent,
    context=[discovery_task]
)


# --------------------------------------------------
# TASK 3: LITERATURE REVIEW
# --------------------------------------------------

literature_review_task = Task(
    description="""
    Prepare a comparative literature review based on
    the discovered papers and their summaries.

    Compare the papers based on:
    1. Research objectives
    2. Methodologies and algorithms
    3. Datasets, if reported
    4. Findings and contributions
    5. Limitations

    Identify similarities and differences between studies.

    Support every comparison using the supplied papers.
    Do not invent experimental results or citations.
    """,

    expected_output="""
    A structured comparative literature review explaining
    the contributions, methodologies, findings and limitations
    of the selected research papers.
    """,

    agent=literature_review_agent,
    context=[discovery_task, summarization_task]
)


# --------------------------------------------------
# TASK 4: RESEARCH GAP IDENTIFICATION
# --------------------------------------------------

research_gap_task = Task(
    description="""
    Identify potential research gaps in the selected research
    topic: {research_topic}

    Use the literature review and paper summaries to identify:
    1. Common limitations across existing studies
    2. Research areas that need further investigation
    3. Methodological limitations
    4. Unexplored or insufficiently explored areas
    5. Possible future research directions

    For every proposed research gap:
    - Explain the supporting evidence.
    - Mention the relevant paper or papers.
    - Distinguish evidence-based gaps from suggestions
      that need further validation.

    Do not claim that a gap is completely unexplored
    based only on a small collection of papers.
    """,

    expected_output="""
    A list of potential research gaps with explanations,
    supporting paper references and suggested future
    research directions.
    """,

    agent=research_gap_agent,
    context=[
        discovery_task,
        summarization_task,
        literature_review_task
    ]
)


# --------------------------------------------------
# TASK 5: FINAL RESEARCH REPORT
# --------------------------------------------------

report_task = Task(
    description="""
    Prepare a comprehensive academic research report
    for the topic: {research_topic}

    Include the following sections:

    1. Research Topic
    2. Introduction
    3. Overview of Selected Research Papers
    4. Individual Paper Summaries
    5. Comparative Literature Review
    6. Identified Potential Research Gaps
    7. Future Research Directions
    8. References with DOI and source links

    Use the outputs of the previous four tasks.

    Ensure that paper titles, authors, years and DOI details
    match the supplied records.

    Do not fabricate references, findings or citations.
    Clearly mention when information is unavailable.
    Write in formal academic language.
    """,

    expected_output="""
    A well-organized academic research report containing
    the paper details, summaries, comparative literature
    review, evidence-based potential research gaps,
    future directions and references.
    """,

    agent=report_agent,
    context=[
        discovery_task,
        summarization_task,
        literature_review_task,
        research_gap_task
    ]
)


# --------------------------------------------------
# DISPLAY TASK INFORMATION
# --------------------------------------------------

if __name__ == "__main__":

    tasks = [
        discovery_task,
        summarization_task,
        literature_review_task,
        research_gap_task,
        report_task
    ]

    print("\nCREWAI RESEARCH ASSISTANT - TASKS")
    print("=" * 45)

    for number, task in enumerate(tasks, start=1):
        print(f"\nTask {number}: {task.agent.role}")
        print(f"Expected output: {task.expected_output[:100]}...")

    print("\nTotal tasks created:", len(tasks))
    print("Task configuration completed successfully.")
