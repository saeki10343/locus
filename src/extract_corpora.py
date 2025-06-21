import json
import re
from tqdm import tqdm
import nltk
# python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

def extract_nl(hunk):
    # コメント・英語的単語の抽出
    words = re.findall(r'//.*|/\*.*?\*/|#.*|\"[^\"]+\"|\'[^\']+\'', hunk, re.DOTALL)
    joined = " ".join(words)
    tokens = word_tokenize(joined.lower())
    return [w for w in tokens if w.isalpha() and w not in STOPWORDS]

def extract_ce(hunk):
    identifiers = set()
    for line in hunk.splitlines():
        if not line.startswith('+'):
            continue
        # 関数・クラス・変数らしきもの
        matches = re.findall(r'\b([a-zA-Z_]\w*)\b', line)
        for match in matches:
            if len(match) > 1 and match.lower() not in STOPWORDS:
                identifiers.add(match)
    return list(identifiers)

def main():
    with open("data/hunks.json") as f:
        hunks = json.load(f)

    result = []
    for h in tqdm(hunks, desc="Extracting corpora"):
        hunk_text = h["hunk"]
        nl_tokens = extract_nl(hunk_text)
        ce_tokens = extract_ce(hunk_text)
        result.append({
            "hunk_id": h["hunk_id"],
            "nl": nl_tokens,
            "ce": ce_tokens
        })

    with open("data/hunk_corpus.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved corpus for {len(result)} hunks to data/hunk_corpus.json")

if __name__ == "__main__":
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")
    main()
