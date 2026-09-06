"""Closed-vocabulary Chinese character-to-pinyin labels for the teaching corpus."""

from __future__ import annotations

PUNCTUATION = "。，；"
LEXICON = {
    "1":"yi1","2":"er4","3":"san1","一":"yi1","二":"er4","三":"san1",
    "第":"di4","帧":"zhen1","中":"zhong1","有":"you3","个":"ge4","可":"ke3","见":"jian4",
    "对":"dui4","象":"xiang4","是":"shi4","红":"hong2","色":"se4","的":"de5","小":"xiao3",
    "大":"da4","等":"deng3","圆":"yuan2","形":"xing2","正":"zheng4","方":"fang1","角":"jiao3",
    "矩":"ju3","五":"wu3","边":"bian1","蓝":"lan2","绿":"lv4","黄":"huang2","紫":"zi3",
    "橙":"cheng2","位":"wei4","于":"yu2","左":"zuo3","右":"you4","上":"shang4","下":"xia4",
    "区":"qu1","域":"yu4","侧":"ce4","央":"yang1",
}
INITIALS = ("zh","ch","sh","b","p","m","f","d","t","n","l","g","k","h","j","q","x","r","z","c","s","y","w")
NORMALIZED = {"1":"一","2":"二","3":"三"}


def character_label(char: str) -> dict[str, object]:
    if char in PUNCTUATION:
        return {"normalized_char": char, "pinyin": "<sil>", "initial": "", "final": "", "tone": 0, "is_punctuation": True}
    if char not in LEXICON:
        raise ValueError(f"Character {char!r} is outside the P1-W05 pronunciation lexicon")
    pinyin = LEXICON[char]
    tone = int(pinyin[-1])
    base = pinyin[:-1]
    initial = next((item for item in INITIALS if base.startswith(item)), "")
    final = base[len(initial):]
    return {"normalized_char": NORMALIZED.get(char, char), "pinyin": pinyin, "initial": initial, "final": final, "tone": tone, "is_punctuation": False}


def validate_text(text: str) -> None:
    if not text:
        raise ValueError("Speech text must be non-empty")
    for char in text:
        character_label(char)
