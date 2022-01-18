import json
import sys

delimiter = " ||| "

for line in sys.stdin:
    try:
        line = json.loads(line)
        src_text, trg_text = line.get("src_text"), line.get("trg_text")
        src_text = src_text.replace("|||", " ").strip()
        trg_text = trg_text.replace("|||", " ").strip()
        if src_text and trg_text:
            print(f"{src_text}{delimiter}{trg_text}")
    except Exception:
        continue
