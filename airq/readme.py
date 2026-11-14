"""Small helper for keeping generated sections of the README up to date."""
import re


def replace_block(text: str, name: str, content: str) -> str:
    """Replace what is between <!-- NAME:START --> and <!-- NAME:END --> markers."""
    pattern = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"README has no <!-- {name}:START --> ... <!-- {name}:END --> block")
    return pattern.sub(lambda m: f"{m.group(1)}\n{content.strip()}\n{m.group(3)}", text, count=1)
