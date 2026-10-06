import os
import re
import time
import random

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {'User-Agent': 'Sainyx/1.0 (text dataset builder)'}

# Wikipedia articles end with link lists and citations. They are noise for a
# language model, so each article is cut at the first of these headings.
TAIL_HEADINGS = {
    "References", "External links", "Further reading", "See also",
    "Notes", "Footnotes", "Bibliography", "Sources",
}

_seen_pages = set()   # canonical titles already scraped, so redirects never add an article twice
_failed = []


def clean_extract(text):
    """Drop the reference/link sections at the end and squeeze blank lines."""
    kept, size = [], 0
    for line in text.split("\n"):
        if line.strip() in TAIL_HEADINGS and size > 200:
            break
        kept.append(line)
        size += len(line) + 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def scrape_wiki(topic, retries=3):
    params = {
        'action': 'query',
        'titles': topic.replace('%27', "'").replace('_', ' '),
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain',
        'redirects': 1,          # follow redirects, otherwise a redirect page returns an empty extract
        'format': 'json',
    }

    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                for page in r.json()['query']['pages'].values():
                    if 'missing' in page:
                        print(f"❌ No such article: {topic}")
                        _failed.append(topic)
                        return ""          # retrying will not help
                    title = page.get('title', topic)
                    if title in _seen_pages:
                        print(f"↩️  Already have: {title}")
                        return ""
                    extract = clean_extract(page.get('extract', ''))
                    if len(extract) > 100:
                        _seen_pages.add(title)
                        print(f"✅ Scraped: {title} ({len(extract):,} chars)")
                        return extract + "\n\n"

            wait = (attempt + 1) * 2
            print(f"⏳ Retrying {topic} in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {topic} — {e}")
            time.sleep(3)

    print(f"❌ Failed: {topic}")
    _failed.append(topic)
    return ""


def category_titles(category, limit=40):
    """Article titles listed directly in a Wikipedia category (subcategories are skipped)."""
    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': category,
        'cmtype': 'page',
        'cmlimit': limit,
        'format': 'json',
    }
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            titles = [m['title'] for m in r.json()['query']['categorymembers']]
            print(f"📂 {category}: {len(titles)} articles")
            return titles
    except Exception as e:
        print(f"⚠️ Category failed: {category} — {e}")
    print(f"📂 {category}: 0 articles (check the category name)")
    return []


# ── Topics ────────────────────────────────────────
dragonball_topics = [
    "Dragon_Ball", "Dragon_Ball_Z", "Dragon_Ball_Super", "Dragon_Ball_GT",
    "Dragon_Ball_(manga)", "Dragon_Ball_Z:_Kakarot", "Dragon_Ball_FighterZ",
    "Goku", "Vegeta", "Gohan", "Piccolo", "Frieza", "Trunks_(Dragon_Ball)",
    "Krillin", "Bulma", "Master_Roshi", "Yamcha", "Tien_Shinhan",
    "Android_18", "Videl", "Goten", "Whis_(Dragon_Ball)", "Zamasu",
    "Cell_(Dragon_Ball)", "Majin_Buu", "Broly_(Dragon_Ball)",
    "Super_Saiyan", "Ultra_Instinct", "Spirit_Bomb",
    "Beerus", "Jiren_(Dragon_Ball)", "Tournament_of_Power",
    "Akira_Toriyama", "Dragon_Ball_Z:_Budokai_Tenkaichi",
]
dragonball_categories = [("Category:Dragon Ball characters", 40), ("Category:Dragon Ball video games", 25)]

anime_topics = [
    "Naruto", "Sasuke_Uchiha", "Kakashi_Hatake", "Itachi_Uchiha", "Boruto:_Naruto_Next_Generations",
    "One_Piece", "Monkey_D._Luffy", "Roronoa_Zoro", "Nami_(One_Piece)",
    "Attack_on_Titan", "Eren_Yeager", "Levi_Ackerman", "Mikasa_Ackerman",
    "Demon_Slayer:_Kimetsu_no_Yaiba", "Tanjiro_Kamado", "Nezuko_Kamado",
    "My_Hero_Academia", "Izuku_Midoriya", "All_Might", "Katsuki_Bakugo",
    "Bleach_(manga)", "Ichigo_Kurosaki", "Hunter_×_Hunter", "Fullmetal_Alchemist",
    "Death_Note", "Jujutsu_Kaisen", "Yuji_Itadori", "Satoru_Gojo",
    "Chainsaw_Man", "Sword_Art_Online", "One-Punch_Man", "Saitama_(One-Punch_Man)",
    "Cowboy_Bebop", "Neon_Genesis_Evangelion", "Fairy_Tail", "Black_Clover",
    "Tokyo_Ghoul", "Vinland_Saga", "Berserk_(manga)", "JoJo%27s_Bizarre_Adventure",
    "Anime", "Manga", "Studio_Ghibli", "Hayao_Miyazaki",
]
anime_categories = [
    ("Category:Naruto characters", 30), ("Category:One Piece characters", 30),
    ("Category:Attack on Titan characters", 25), ("Category:My Hero Academia characters", 25),
]

gaming_topics = [
    "Grand_Theft_Auto_V", "God_of_War_(2018_video_game)", "God_of_War_Ragnarök",
    "Devil_May_Cry", "Dark_Souls", "Dark_Souls_III", "Elden_Ring",
    "Call_of_Duty", "Mortal_Kombat", "Street_Fighter", "Tekken",
    "Red_Dead_Redemption_2", "The_Witcher_3:_Wild_Hunt", "Sekiro:_Shadows_Die_Twice",
    "Doom_(franchise)", "Halo_(franchise)", "Bloodborne", "Hollow_Knight",
    "The_Legend_of_Zelda:_Breath_of_the_Wild", "Resident_Evil", "Metal_Gear_Solid",
    "Final_Fantasy_VII", "Persona_5", "Minecraft", "Fortnite", "Cyberpunk_2077",
    "Assassin%27s_Creed", "Batman:_Arkham_Asylum", "Nier:_Automata", "Bayonetta",
    "Video_game", "Role-playing_video_game", "Fighting_game",
]
gaming_categories = [("Category:Fighting games", 25), ("Category:Action role-playing video games", 25)]

story_topics = [
    "Horror_fiction", "Action_fiction", "Supernatural_horror", "Hero%27s_journey",
    "Shōnen_manga", "Fantasy_literature", "Science_fiction", "Apocalyptic_fiction",
    "Dark_fantasy", "Cyberpunk", "Post-apocalyptic_fiction", "Superhero_fiction",
    "Martial_arts_film", "Anti-hero", "Plot_(narrative)", "Character_(arts)",
    "Narrative", "Storytelling", "Worldbuilding", "Mythology",
]
story_categories = []


def build_dataset(topics, categories, filename):
    print(f"\nBuilding {filename}...")
    titles = list(topics)
    for category, limit in categories:
        titles += category_titles(category, limit)
        time.sleep(random.uniform(1.0, 2.0))

    content = ""
    for topic in titles:
        content += scrape_wiki(topic)
        # random delay to avoid rate limiting
        time.sleep(random.uniform(1.5, 3.0))

    with open(os.path.join(SCRIPT_DIR, filename), 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Saved {filename} — {len(content):,} characters")
    return content


# ── Run ───────────────────────────────────────────
if __name__ == "__main__":
    all_content = ""
    all_content += build_dataset(dragonball_topics, dragonball_categories, "dragonball.txt")
    all_content += build_dataset(anime_topics, anime_categories, "anime.txt")
    all_content += build_dataset(gaming_topics, gaming_categories, "gaming.txt")
    all_content += build_dataset(story_topics, story_categories, "stories.txt")

    with open(os.path.join(SCRIPT_DIR, "sainyx_data.txt"), 'w', encoding='utf-8') as f:
        f.write(all_content)

    print(f"\n🔥 Sainyx dataset ready!")
    print(f"Articles scraped: {len(_seen_pages)}")
    if _failed:
        print(f"Not scraped ({len(_failed)}): {', '.join(_failed)}")
    print(f"Total size: {len(all_content):,} characters")