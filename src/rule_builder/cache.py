import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("configs/rules/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(text):

    hash_name = hashlib.md5(text.encode()).hexdigest()

    return CACHE_DIR / f"{hash_name}.json"


def load_cache(text):

    path = get_cache_path(text)

    if path.exists():

        with open(path) as f:

            return json.load(f)

    return None


def save_cache(text, data):

    path = get_cache_path(text)

    with open(path, "w") as f:

        json.dump(data, f, indent=2)