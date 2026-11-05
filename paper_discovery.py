
import os
import requests
from dotenv import load_dotenv

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)


def reconstruct_abstract(inverted_index):

    if not inverted_index:
        return "Abstract not available."

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda x: x[0])

    return " ".join(word for _, word in words)


def discover_papers(topic, number_of_papers=5, year_from=2020):

    url = "https://api.openalex.org/works"

    params = {
        "search": topic,
        "filter": f"from_publication_date:{year_from}-01-01",
        "sort": "relevance_score:desc",
        "per_page": number_of_papers,
    }

    email = os.getenv("OPENALEX_EMAIL")

    if email:
        params["mailto"] = email

    try:

        print("Connecting to OpenAlex...")

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        papers = []

        for item in data.get("results", []):

            authors = [
                author.get("author", {}).get("display_name")
                for author in item.get("authorships", [])
                if author.get("author", {}).get("display_name")
            ]

            primary_location = item.get("primary_location") or {}
            source = primary_location.get("source") or {}

            paper = {
                "title": item.get("title") or "Title unavailable",
                "authors": authors,
                "year": item.get("publication_year"),
                "doi": item.get("doi") or "DOI unavailable",
                "abstract": reconstruct_abstract(
                    item.get("abstract_inverted_index")
                ),
                "venue": source.get("display_name") or "Venue unavailable",
                "source_url": (
                    primary_location.get("landing_page_url")
                    or item.get("id")
                )
            }

            papers.append(paper)

        return papers

    except requests.RequestException as error:

        print("OpenAlex connection error:", error)

        return []


# TEST THE DISCOVERY AGENT DATA SOURCE

if __name__ == "__main__":

    topic = "Generative AI in Healthcare"

    print("\nRESEARCH PAPERS DISCOVERED")
    print("=" * 60)

    papers = discover_papers(
        topic=topic,
        number_of_papers=5,
        year_from=2020
    )

    if not papers:
        print("No papers found. Check internet connection or search topic.")

    for index, paper in enumerate(papers, start=1):

        print(f"\nPaper {index}")
        print("Title:", paper["title"])
        print("Authors:", ", ".join(paper["authors"]))
        print("Year:", paper["year"])
        print("DOI:", paper["doi"])
        print("Journal:", paper["venue"])
        print("Source:", paper["source_url"])
        print("Abstract:", paper["abstract"][:500])

        print("-" * 60)

    print("\nTotal papers retrieved:", len(papers))
