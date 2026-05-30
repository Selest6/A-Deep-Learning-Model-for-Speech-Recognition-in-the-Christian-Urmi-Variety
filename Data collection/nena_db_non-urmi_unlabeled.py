import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tempfile
import os
from pydub import AudioSegment
from openpyxl import Workbook



ffmpeg_bin = r"C:\Users\alesi\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_bin
AudioSegment.converter = os.path.join(ffmpeg_bin, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_bin, "ffprobe.exe")


base_folder = os.path.dirname(os.path.abspath(__file__))
unlabeled_audios_folder = os.path.join(base_folder, "non-urmi_unlabeled_data", "audios")

os.makedirs(unlabeled_audios_folder, exist_ok=True)


no_tr_excel_path = os.path.join(base_folder, "non-urmi_no_transcript_metadata.xlsx")
wb_no_tr = Workbook()
ws_no_tr = wb_no_tr.active
ws_no_tr.title = "metadata"

unlabeled_headers = [
    "title of the text",
    "audio file",
    "audio file length (seconds)",
    "source",
    "dialect"
]


ws_no_tr.append(unlabeled_headers)


def extract_title_and_dialect(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    h1 = soup.find("h1")
    if not h1:
        return None, None

    text = h1.get_text(strip=True)

    if ":" in text:
        dialect, title = text.split(":", 1)
        return title.strip(), dialect.strip()
    else:
        return text.strip(), None


BASE = "https://nena.ames.cam.ac.uk"
LIST_URL = "https://nena.ames.cam.ac.uk/dialects/?community=C&location="


session = requests.Session()

session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
})

session.cookies.update({
    "sessionid": "xfp9tqx8fccsr9grlp9bqifdccfe3wel",
    "csrftoken": "MMZlOcypFeTXWfnoBPhuT3L11YaO0RAEjgliTZzRzUHZhY1K30px88YQYrnv7f0m"
})


check = session.get("https://nena.ames.cam.ac.uk/dialects/")
if "Log out" in check.text or "Logout" in check.text:
    print("SESSION AUTHORIZED")
else:
    print("LOGIN FAILED")


r = session.get(LIST_URL)
soup = BeautifulSoup(r.text, "html.parser")

dialect_links = {}

table = soup.find("table")
rows = table.find_all("tr")[1:]

for row in rows:
    first_col = row.find("td")
    a = first_col.find("a")

    name = a.get_text(strip=True)
    link = urljoin(BASE, a["href"])

    dialect_links[name] = link

print(f"Collected dialects: {len(dialect_links)}")


audio_dict = {}

for name, link in dialect_links.items():
    page = session.get(link)
    s = BeautifulSoup(page.text, "html.parser")

    audio_a = s.find("a", string=lambda x: x and "Browse audio and transcripts" in x)

    if audio_a:
        audio_link = urljoin(BASE, audio_a["href"])
        audio_dict[name] = audio_link

audio_dict.pop("Urmi, Christian", None)

print(f"Dialects with audio: {len(audio_dict)}")
for k, v in audio_dict.items():
    print(f"{k}: {v}")


unlabeled_counter = 1

for _, main_url in audio_dict.items():

    soup_main = BeautifulSoup(
        session.get(main_url).content,
        "html.parser"
    )

    audio_page_links = sorted({
        "https://nena.ames.cam.ac.uk" + a["href"]
        for a in soup_main.find_all("a", href=True)
        if a["href"].startswith("/audio/")
    })

    audio_page_links = [
        url for url in audio_page_links
        if url.rstrip("/") != "https://nena.ames.cam.ac.uk/audio"
    ]

    bad_urls = ["https://nena.ames.cam.ac.uk/audio/10/", "https://nena.ames.cam.ac.uk/audio/62/"]

    for bad_url in bad_urls:
        if bad_url in audio_page_links:
            audio_page_links.remove(bad_url)

    print(f"Found audios: {len(audio_page_links)}")

    for page_url in audio_page_links:
        print(f"{page_url}")

        soup = BeautifulSoup(
            session.get(page_url).content,
            "html.parser"
        )

        title_text, dialect_name = extract_title_and_dialect(soup)

        audio_link = soup.find("a", href=lambda x: x and x.endswith(".mp3"))
        if not audio_link:
            continue

        mp3_url = audio_link["href"]
        if not mp3_url.startswith("http"):
            mp3_url = "https://nena.ames.cam.ac.uk" + mp3_url

        unlabeled_wav_path = os.path.join(
            unlabeled_audios_folder,
            f"audio{unlabeled_counter}_non-urmi_unlabeled.wav"
        )


        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(session.get(mp3_url).content)
            tmp_path = tmp.name

        sound = AudioSegment.from_mp3(tmp_path)
        sound.export(unlabeled_wav_path, format="wav")
        os.remove(tmp_path)

        total_seconds = int(sound.duration_seconds)
        audio_length = f"{total_seconds//3600:02d}:{(total_seconds%3600)//60:02d}:{total_seconds%60:02d}"


        ws_no_tr.append([
            title_text,
            os.path.basename(unlabeled_wav_path),
            audio_length,
            "The North-Eastern Neo-Aramaic Database Project",
            dialect_name
        ])

        unlabeled_counter += 1

wb_no_tr.save(no_tr_excel_path)
