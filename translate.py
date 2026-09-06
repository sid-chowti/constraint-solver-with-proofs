"""English puzzle text -> a Puzzle and its clues, using an AI to read the words.

The AI's ONLY job here is language. It never solves anything, never decides
what follows from what, and never sees the grid. It reads sentences and writes
down what they say; the solver does every piece of reasoning.

That split is the whole point of the project, and it is enforced by what comes
back: structured data, checked by three guards before the solver ever sees it.

    text -> AI -> JSON -> clues_from_json -> validate_constraints -> solver
                          "is this a clue?"  "does the puzzle have
                                              what it names?"

When a guard objects, its complaint goes back to the AI and it tries again.
Plain Python is the judge; the AI only proposes.

Testing note: translate() takes an `ask` function rather than calling the API
itself. Every retry path can then be exercised with canned replies and no API
key, which is what makes the interesting half of this file testable at all.
"""
import json
import re

from constraints import InvalidConstraint, validate_constraints
from grounding import complaints as grounding_complaints
from parsing import UnreadableClues, clues_from_json
from puzzle import Puzzle

MODEL = "claude-opus-5"

# Three is enough: the guards report EVERY problem at once, so a second attempt
# already has the full list. A third covers one bad round; more just burns money.
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You translate logic puzzles from English into JSON. You do NOT solve them.

Never work out the answer. Never leave a clue out because it seems redundant.
Never invent a clue that the text does not state. Your only job is to write
down what each sentence says.

Reply with ONE JSON object and nothing else:

{
  "num_positions": 5,
  "categories": {
    "nation":  ["English", "Spaniard", "Ukrainian", "Norwegian", "Japanese"],
    "color":   ["red", "green", "ivory", "yellow", "blue"]
  },
  "clues": [ ... ]
}

Rules for the puzzle itself:
- Every category must list EXACTLY num_positions values.
- Positions are numbered from 1. Position 1 is the leftmost / first.
- Category and value names must be the words the puzzle uses.

Clue types:

1. A thing at a literal position:
   {"type": "AbsolutePosition", "category_value": ["nation", "Norwegian"],
    "operator": "==", "position": 1, "source_text": "<the original sentence>"}

   Operators: "==" (is at), "!=" (is not at), "<" (is left of / before),
   ">" (is right of / after), "<=", ">=".

2. Two things compared to each other:
   {"type": "RelativePosition", "a": ["color", "green"], "b": ["color", "ivory"],
    "operator": "==", "offset": 1, "source_text": "<the original sentence>"}

   This means:  position(a) + offset  <operator>  position(b)

   So:
   - same position ("the Spaniard owns the dog"):  offset 0, operator "=="
   - a immediately LEFT of b:                      offset 1, operator "=="
   - a immediately RIGHT of b:                     offset -1, operator "=="
   - a somewhere left of b:                        offset 0, operator "<"
   - a somewhere right of b:                       offset 0, operator ">"

3. Several things that must ALL hold:
   {"type": "And", "constraints": [ ... ], "source_text": "..."}

4. Several things where AT LEAST ONE holds:
   {"type": "Or", "constraints": [ ... ], "source_text": "..."}

   Use Or for "next to" (immediately left OR immediately right) and for
   "at one end" (first position OR last position).

Every clue must carry "source_text": the puzzle's own sentence, word for word.
It is quoted back to the reader in the explanation, so copy it exactly.

Reply with the JSON object only. No commentary, no markdown fences."""


class TranslationFailed(Exception):
    """The AI could not produce usable clues within the attempt limit.

    Carries `attempts`, the complaint list from each try, so a caller can show
    the user what went wrong rather than a bare failure.
    """

    def __init__(self, attempts):
        self.attempts = attempts
        super().__init__(
            f"could not translate the puzzle after {len(attempts)} attempt(s); "
            f"last problems: {attempts[-1] if attempts else 'none'}"
        )


def translate(text, ask, max_attempts=MAX_ATTEMPTS):
    """Read an English puzzle into a (Puzzle, clues) pair.

    `ask` takes the running conversation and returns the model's reply as text.
    Keeping it a parameter is what lets every branch below be tested without an
    API key.

    Raises TranslationFailed if no attempt gets past the guards.
    """
    conversation = [{"role": "user", "content": _first_request(text)}]
    attempts = []

    for _ in range(max_attempts):
        reply = ask(conversation)
        puzzle, clues, complaints = _read_reply(reply, text)

        if not complaints:
            return puzzle, clues

        attempts.append(complaints)

        # Hand the reply and the objections back so the next try can fix them.
        # Every guard reports ALL its problems, so this is the full list.
        conversation = conversation + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": _retry_request(complaints)},
        ]

    raise TranslationFailed(attempts)


def _read_reply(reply, text=""):
    """Turn one model reply into (puzzle, clues, complaints).

    Complaints being empty is the only success. Everything else hands back a
    list of plain sentences for the AI to act on.

    `text` is the user's original puzzle, needed only by the grounding guard.
    """
    data = _extract_json(reply)
    if data is None:
        return None, None, ["Your reply was not a single JSON object. "
                            "Reply with only the JSON, no commentary."]

    puzzle, complaints = _read_puzzle(data)
    if puzzle is None:
        return None, None, complaints

    try:
        clues = clues_from_json(data.get("clues"))
    except UnreadableClues as broken:
        return None, None, [f"{p.where}: {p.problem} (expected one of: {p.allowed})"
                            for p in broken.problems]

    # The clues are well-formed; do they name things this puzzle actually has?
    try:
        validate_constraints(puzzle, clues)
    except InvalidConstraint as wrong:
        return None, None, [str(wrong)]

    # Both guards above check the AI against ITSELF, and an invented value is
    # perfectly consistent with the rest of an invented puzzle. This one checks
    # it against the user: every value should be traceable to their own words.
    invented = grounding_complaints(puzzle.categories, text)
    if invented:
        return None, None, invented

    return puzzle, clues, []


# The same value in two categories. Almost always the AI confusing itself, but
# it CAN be legitimate ("Green" as both a colour and a surname), so this is a
# warning shown on the confirmation screen and never a reason to retry -
# retrying on a legitimate puzzle would loop until it gave up.
def duplicate_values(categories):
    seen = {}
    for name, values in categories.items():
        for value in values:
            seen.setdefault(value, []).append(name)
    return {value: names for value, names in seen.items() if len(names) > 1}


def warnings_for(puzzle):
    """Things worth a person's attention that are not worth refusing over."""
    return [
        f"'{value}' is a value in more than one category ({', '.join(names)}) "
        f"- check that is really what your puzzle means"
        for value, names in sorted(duplicate_values(puzzle.categories).items())
    ]


def _read_puzzle(data):
    """The categories and size. Returns (puzzle, complaints); one of them is
    always empty."""
    if not isinstance(data, dict):
        return None, ["The top level of your reply must be a JSON object."]

    size = data.get("num_positions")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        return None, [f"'num_positions' must be a whole number of at least 1, got {size!r}."]

    categories = data.get("categories")
    if not isinstance(categories, dict) or not categories:
        return None, ["'categories' must be a JSON object of category name -> list of values."]

    for name, values in categories.items():
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            return None, [f"Category '{name}' must be a list of value names as strings."]

    # Puzzle checks the rest itself: right number of values, no duplicates.
    try:
        return Puzzle(categories, size), []
    except ValueError as bad:
        return None, [str(bad)]


def _extract_json(reply):
    """Pull the JSON object out of a reply, or None.

    The prompt asks for bare JSON, but models often wrap it in a markdown
    fence anyway, so both are accepted. Being lenient about the WRAPPER is
    fine — nothing about the clues themselves is being guessed at.
    """
    if not isinstance(reply, str):
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", reply, re.DOTALL)
    candidate = fenced.group(1) if fenced else reply

    # Fall back to the outermost braces, in case of stray text either side.
    if not fenced:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = candidate[start:end + 1]

    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return None


def _first_request(text):
    return f"Translate this logic puzzle into JSON.\n\n{text}"


def _retry_request(complaints):
    """The objections, as instructions. Numbered so nothing gets skimmed."""
    listed = "\n".join(f"{number}. {complaint}"
                       for number, complaint in enumerate(complaints, 1))
    return (
        "That reply could not be used. Problems found:\n\n"
        f"{listed}\n\n"
        "Fix every one of them and reply with the corrected JSON object only."
    )


def anthropic_asker(api_key, model=MODEL):
    """An `ask` backed by the real Claude API.

    Imported lazily so the solver, the tests and the example puzzles all work
    with the anthropic package absent — only this path needs it.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def ask(conversation):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            # Translation is the one step whose mistakes nothing downstream can
            # catch: a misread clue becomes a valid puzzle with a wrong answer.
            # Worth letting the model think about it.
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=conversation,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return ask
