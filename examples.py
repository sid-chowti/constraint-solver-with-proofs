"""The bundled example puzzle, already translated.

The one example ships as finished JSON rather than English, so the site can
solve it with no API key and no cost to anyone. The AI translator is only
needed when a visitor types a puzzle of their own.

It is stored in exactly the format the translator produces, so it is also a
worked example of what the AI is being asked for — and it round-trips through
the same parser, which means a broken example fails a test rather than a demo.
"""
import json
import pathlib

from parsing import clues_from_json
from puzzle import Puzzle

EXAMPLES = pathlib.Path(__file__).parent / "examples"

# The order they are offered in: the famous one leads, then the gentle one,
# then the rest. Anything not listed follows alphabetically.
ORDER = ["einstein", "desks", "trucks"]


def available():
    """Every bundled puzzle, as small summaries for the page to offer.

    Also the allow-list: the name arrives from a query string, so nothing may
    reach the filesystem without appearing here first.
    """
    found = []
    for path in EXAMPLES.glob("*.json"):
        if path.stem.endswith("_prose"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        found.append({"name": path.stem, "title": data["name"],
                      "question": data["question"],
                      "positions": data["num_positions"],
                      "clues": len(data["clues"])})

    def rank(example):
        name = example["name"]
        return (ORDER.index(name) if name in ORDER else len(ORDER), name)

    return sorted(found, key=rank)


def load(name="einstein"):
    """Read a bundled example. Returns (info, puzzle, clues), where `info`
    carries the display name, the original English text, and the question."""
    data = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))

    puzzle = Puzzle(data["categories"], data["num_positions"])
    clues = clues_from_json(data["clues"])

    info = {"name": data["name"], "text": data["text"], "question": data["question"],
            # What one position is called here: a house, a desk, a food truck.
            # Calling a food truck a house is a small thing that reads as a bug.
            "position_noun": data.get("position_noun", "position")}
    return info, puzzle, clues


def load_prose(name="einstein"):
    """The stored plain-English wording for a bundled puzzle, keyed by step id,
    or None if this example has none.

    Written once and checked in, so the demo reads like English without an API
    key and without costing anyone anything. narrate.problems_with is what
    guarantees it still lines up with the trace - see test_examples.
    """
    path = EXAMPLES / f"{name}_prose.json"
    if not path.exists():
        return None

    stored = json.loads(path.read_text(encoding="utf-8"))["prose"]
    return {int(step): sentence for step, sentence in stored.items()}
