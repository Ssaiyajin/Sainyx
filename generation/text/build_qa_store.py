"""Build data/text/qa_store.json from the Wikipedia intro of every dataset topic.

Run from the repo root with internet access:  python -m generation.text.build_qa_store
Every answer is a sentence taken from Wikipedia, so nothing is generated or invented.
"""
import json
import os
import time
import urllib.parse
from collections import Counter

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


def fetch_leads(titles, get=requests.get, sleep=time.sleep):
    """Return ({page title: intro text}, {page title: [redirect names]}).

    Pages reached through a redirect to a *section* (Gogeta -> List of Dragon Ball characters#Gogeta)
    are dropped: the page intro would describe the whole list, not the character.
    """
    leads, aliases, covered, skip = {}, {}, set(), set()
    status = Counter()

    def api_get(params, tries=5):
        for k in range(tries):
            r = get(bd.API_URL, headers=bd.HEADERS, timeout=30, params=params)
            status[r.status_code] += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                sleep(3 * (k + 1))
                continue
            return None
        return None

    for i in range(0, len(titles), 20):
        params = {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": "max",
                  "redirects": 1, "titles": "|".join(titles[i:i + 20]), "format": "json"}
        while True:
            data = api_get(params)
            if not data:
                break
            q = data.get("query", {})
            for page in q.get("pages", {}).values():
                covered.add(page.get("title", "").lower())
                if page.get("extract") and "missing" not in page:
                    leads.setdefault(page["title"], page["extract"])
            for m in q.get("normalized", []):
                covered.add(m["from"].lower())
            for m in q.get("redirects", []):
                covered.add(m["from"].lower())
                if m.get("tofragment"):
                    skip.add(m["to"])
                else:
                    aliases.setdefault(m["to"], []).append(m["from"])
            if "continue" in data:
                params = {**params, **data["continue"]}
            else:
                break
        sleep(1.0)
    for t in skip - set(titles):
        leads.pop(t, None)
    print("batch pass:", len(leads), "leads | HTTP:", dict(status))

    for t in [t for t in titles if t.lower() not in covered and t not in leads]:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(t.replace(" ", "_"), safe="")
        for k in range(3):
            r = get(url, headers=bd.HEADERS, timeout=30, params={"redirect": "false"})
            if r.status_code == 200:
                j = r.json()
                if j.get("type") == "standard" and j.get("extract"):
                    leads.setdefault(j["title"], j["extract"])
                break
            if r.status_code in (429, 500, 502, 503, 504):
                sleep(3 * (k + 1))
                continue
            break
        sleep(0.3)
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