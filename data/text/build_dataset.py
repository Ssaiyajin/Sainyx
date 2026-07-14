import requests
import time
import random

def scrape_wiki(topic, retries=3):
    headers = {'User-Agent': 'Sainyx/1.0'}
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'titles': topic.replace('%27', "'").replace('_', ' '),
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain',
        'format': 'json'
    }

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                pages = r.json()['query']['pages']
                for page in pages.values():
                    if 'extract' in page and len(page['extract']) > 100:
                        print(f"✅ Scraped: {topic} ({len(page['extract']):,} chars)")
                        return page['extract'] + "\n\n"
            
            # wait before retry
            wait = (attempt + 1) * 2
            print(f"⏳ Retrying {topic} in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {topic} — {e}")
            time.sleep(3)

    print(f"❌ Failed: {topic}")
    return ""

# ── Topics ────────────────────────────────────────
dragonball_topics = [
    "Dragon_Ball", "Dragon_Ball_Z", "Dragon_Ball_Super",
    "Goku", "Vegeta", "Gohan", "Piccolo", "Frieza",
    "Cell_(Dragon_Ball)", "Majin_Buu", "Broly_(Dragon_Ball)",
    "Super_Saiyan", "Ultra_Instinct", "Spirit_Bomb",
    "Beerus", "Jiren_(Dragon_Ball)", "Tournament_of_Power"
]

anime_topics = [
    "Naruto", "Sasuke_Uchiha", "One_Piece", "Monkey_D._Luffy",
    "Attack_on_Titan", "Eren_Yeager", "Demon_Slayer:_Kimetsu_no_Yaiba",
    "Tanjiro_Kamado", "My_Hero_Academia", "Izuku_Midoriya",
    "Bleach_(manga)", "Ichigo_Kurosaki", "Hunter_×_Hunter",
    "Fullmetal_Alchemist", "Death_Note", "Jujutsu_Kaisen",
    "Chainsaw_Man", "Sword_Art_Online"
]

gaming_topics = [
    "Grand_Theft_Auto_V", "God_of_War_(2018_video_game)",
    "Devil_May_Cry", "Dark_Souls", "Elden_Ring",
    "Call_of_Duty", "Mortal_Kombat", "Street_Fighter",
    "Tekken", "Red_Dead_Redemption_2",
    "The_Witcher_3:_Wild_Hunt", "Sekiro:_Shadows_Die_Twice",
    "Doom_(franchise)", "Halo_(franchise)"
]

story_topics = [
    "Horror_fiction", "Action_fiction",
    "Supernatural_horror", "Hero%27s_journey",
    "Shōnen_manga", "Fantasy_literature",
    "Science_fiction", "Apocalyptic_fiction"
]

def build_dataset(topics, filename):
    print(f"\nBuilding {filename}...")
    content = ""
    for topic in topics:
        content += scrape_wiki(topic)
        # random delay to avoid rate limiting
        time.sleep(random.uniform(1.5, 3.0))
    
    with open(f"data/{filename}", 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Saved {filename} — {len(content):,} characters")
    return content

# ── Run ───────────────────────────────────────────
all_content = ""
all_content += build_dataset(dragonball_topics, "dragonball.txt")
all_content += build_dataset(anime_topics, "anime.txt")
all_content += build_dataset(gaming_topics, "gaming.txt")
all_content += build_dataset(story_topics, "stories.txt")

with open("data/sainyx_data.txt", 'w', encoding='utf-8') as f:
    f.write(all_content)

print(f"\n🔥 Sainyx dataset ready!")
print(f"Total size: {len(all_content):,} characters")