"""Put your deployed dashboard address into the README.

    python tools/add_live_link.py https://your-app-name.streamlit.app
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from airq.readme import replace_block      # noqa: E402


def main() -> None:
    if len(sys.argv) != 2 or not re.match(r"^https://[\w.-]+\.\w+(/\S*)?$", sys.argv[1]):
        sys.exit("Usage: python tools/add_live_link.py https://your-app-name.streamlit.app")
    url = sys.argv[1].rstrip("/")
    readme = ROOT / "README.md"
    text = replace_block(readme.read_text(encoding="utf-8"), "LIVE",
                         f"**Live dashboard:** [{url}]({url})")
    readme.write_text(text, encoding="utf-8")
    print("README.md updated with", url)


if __name__ == "__main__":
    main()
