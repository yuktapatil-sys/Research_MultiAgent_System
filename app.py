
import json
import streamlit as st

from paper_discovery import discover_papers
from crew import run_research


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI-Powered Research Paper Assistant")

st.write(
    "Discover research papers, summarize studies, "
    "prepare literature reviews and identify potential "
    "research gaps using five AI agents."
)

st.divider()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("Research Configuration")

topic = st.text_input(
    "Enter your research topic",
    placeholder="Example: Generative AI in Healthcare"
)

col1, col2 = st.columns(2)

with col1:
    number_of_papers = st.slider(
        "Number of research papers",
        min_value=3,
        max_value=15,
        value=5
    )

with col2:
    year_from = st.selectbox(
        "Search papers published from",
        options=list(range(2026, 2014, -1)),
        index=6
    )

st.caption(
    "Papers are retrieved from OpenAlex. "
    "The selected year is the starting publication year."
)

st.divider()


# --------------------------------------------------
# START RESEARCH
# --------------------------------------------------

if st.button("Start Research", type="primary"):

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        try:

            # PHASE 1: DISCOVER PAPERS

            with st.spinner("Agent 1: Discovering research papers..."):

                papers = discover_papers(
                    topic,
                    number_of_papers=number_of_papers,
                    year_from=year_from
                )

            if not papers:

                st.error(
                    "No research papers were found. "
                    "Try another topic or an earlier year."
                )

            else:

                st.success(
                    f"{len(papers)} research papers retrieved successfully!"
                )

                # DISPLAY RETRIEVED PAPERS

                st.subheader("Retrieved Research Papers")

                for index, paper in enumerate(papers, start=1):

                    with st.expander(
                        f"{index}. {paper.get('title', 'Title unavailable')}"
                    ):

                        st.write(
                            "**Authors:**",
                            paper.get("authors", "Not available")
                        )

                        st.write(
                            "**Publication Year:**",
                            paper.get("year", "Not available")
                        )

                        st.write(
                            "**Journal / Venue:**",
                            paper.get("venue", "Not available")
                        )

                        st.write(
                            "**DOI:**",
                            paper.get("doi", "Not available")
                        )

                        st.write(
                            "**Abstract:**",
                            paper.get("abstract", "Not available")
                        )

                        st.write(
                            "**Source:**",
                            paper.get("source_url", "Not available")
                        )

                # PHASE 2: RUN FIVE AI AGENTS

                st.subheader("AI Agent Processing")

                with st.spinner(
                    "Agents 1–5: Analyzing papers and preparing the report..."
                ):

                    papers_data = json.dumps(
                        papers,
                        ensure_ascii=False,
                        indent=2
                    )

                    result = run_research(
                        topic=topic,
                        papers_data=papers_data
                    )

                    report_text = str(result)

                st.success("Research analysis completed successfully!")

                # PHASE 3: DISPLAY REPORT

                st.divider()

                st.subheader("Final Research Report")

                st.markdown(report_text)

                # DOWNLOAD REPORT

                st.download_button(
                    label="Download Research Report",
                    data=report_text,
                    file_name="AI_Research_Report.txt",
                    mime="text/plain"
                )

        except Exception as error:

            st.error("An error occurred while processing the research.")

            st.exception(error)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Powered Multi-Agent Research Assistant | "
    "OpenAlex + CrewAI + Groq + Streamlit"
)
