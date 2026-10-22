"""Emojis in a reply become body language, not words.

TTS read emojis out ("smiling face with open mouth"). split_emojis() removes
every emoji from the spoken text and maps the ones it knows to an emotion
motion (body/motions.py) for the robot to play while it talks.
"""

import re

EMOJI_EMOTIONS = {
    "happy": "😀😃😄😁😊🙂☺️👍😌😎",
    "excited": "🎉🤩🚀✨🙌🥳💪⚡🔥",
    "laugh": "😂🤣😆😹😅",
    "love": "❤️😍🥰🤗💖💕😘💙💚",
    "sad": "😢😞😔🥺😭☹️🙁💔",
    "surprised": "😮😲🤯😱😯😳‼️",
    "thinking": "🤔🧐💭🤨",
}
_EMOTION_OF = {e: name for name, emojis in EMOJI_EMOTIONS.items()
               for e in re.findall(r".️?", emojis)}

# pictographs, symbols, dingbats, flags, variation selectors, joiners
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "←-⇿⬀-⯿‼⁉️‍]+"
)


def split_emojis(text: str) -> tuple[str, list[str]]:
    """(text without emojis, emotions in order of first appearance, deduped)."""
    emotions: list[str] = []
    for run in _EMOJI.findall(text):
        for emoji in re.findall(r".️?", run):
            emotion = _EMOTION_OF.get(emoji) or _EMOTION_OF.get(emoji.rstrip("️"))
            if emotion and emotion not in emotions:
                emotions.append(emotion)
    clean = " ".join(_EMOJI.sub(" ", text).split())
    clean = re.sub(r"\s+([.,!?;:])", r"\1", clean)
    return clean, emotions
