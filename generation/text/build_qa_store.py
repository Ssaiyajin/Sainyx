"""Build data/text/qa_store.json from the Wikipedia intro of every dataset topic.

Run from the repo root with internet access:  python -m generation.text.build_qa_store
Every answer is a sentence taken from Wikipedia, so nothing is generated or invented.
"""
import json
import os
import time

import requests

import data.text.build_dataset as bd
from generation.text.retrieval import DEFAULT_STORE_PATH, lead_to_sentences, make_answer, short_title


def collect_titles():
    titles = []
    for topics, cats in [(bd.dragonball_topics, bd.dragonball_categories), (bd.anime_topics, bd.anime_categories),
                         (bd.gaming_topics, bd.gaming_categories), (bd.story_topics, bd.story_categories)]:
        titles += [t.replace("%27", "'").replace("_", " ") for t in topics]
        for cat, limit in cats:
            titles += bd.category_titles(cat, limit)
            time.sleep(1.0)
    return list(dict.fromkeys(titles))


def fetch_leads(titles):
    leads, aliases = {}, {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        r = requests.get(bd.API_URL, headers=bd.HEADERS, timeout=30, params={
            "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": "max",
            "redirects": 1, "titles": "|".join(batch), "format": "json"})
        if r.status_code != 200:
            continue
        q = r.json().get("query", {})
        for page in q.get("pages", {}).values():
            if "missing" not in page and page.get("extract"):
                leads[page["title"]] = page["extract"]
        for red in q.get("redirects", []) + q.get("normalized", []):
            aliases.setdefault(red["to"], []).append(red["from"])
        time.sleep(1.0)
    return leads, aliases


def main(path=DEFAULT_STORE_PATH):
    leads, aliases = fetch_leads(collect_titles())
    entries = []
    for title, lead in leads.items():
        sentences = lead_to_sentences(lead)
        answer = make_answer(sentences)
        if not answer:
            continue
        entries.append({"title": title, "aliases": sorted(set(aliases.get(title, [])) - {title, short_title(title)}),
                        "answer": answer, "facts": sentences, "source": "wikipedia"})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "entries": entries}, f, ensure_ascii=False, indent=1)
    print(f"{len(entries)} entries written to {path}")


if __name__ == "__main__":
    main()