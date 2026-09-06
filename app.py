"""The web front door: a page, an example, and a two-step solve.

Deliberately thin. Every hard thing already happened underneath — this layer
only moves data in and out, so nothing here decides anything about a puzzle.

Three ways in:

    GET  /api/example    the bundled Einstein puzzle, already translated and
                         solved. No API key, no cost, works for everyone. It
                         skips the confirmation step below on purpose: a human
                         already checked it, so there is no AI reading to
                         second-guess.

    POST /api/translate  English, plus the VISITOR'S OWN API key. Returns the
                         puzzle and clues the AI read out of it, each said back
                         in plain words. Solves NOTHING.

    POST /api/solve      those clues back again. Returns the answer and the
                         explanation. No AI, no key, no cost.

WHY TWO STEPS. No code can catch the AI *consistently* misreading a sentence.
If it reverses "green is immediately left of white", every guard passes and the
solver returns a guaranteed-correct answer to the wrong puzzle. Catching that
needs understanding English — the exact job the AI was hired for, so it cannot
also be the check. Only a person can. The screen between these two endpoints is
where that person looks, and it comes BEFORE any answer exists, so there is
nothing to anchor on and rubber-stamp.

WHY THE CLUES TRAVEL BACK UP rather than waiting here between the two calls:
nothing on this server remembers anything between requests. A pending list
would need expiry, would leak every abandoned puzzle, and would lose them all
on restart — and a free host sleeps whenever nobody is visiting, which is
exactly while somebody is reading the screen. So the browser carries them.

That is safe because /api/solve trusts nothing it is handed. The same three
guards that never trusted the AI do not care that the data arrived from a
browser this time. And a visitor rewriting their own puzzle is the feature, not
an attack — there is no privilege here to escalate, only their own puzzle to
change.
"""
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from constraints import InvalidConstraint
from deduction import to_ai_payload
from examples import load
from parsing import UnreadableClues, clue_to_json, clues_from_json
from puzzle import Puzzle
from solve import solve
from translate import TranslationFailed, anthropic_asker, translate

app = FastAPI(title="Logic Puzzle Solver")

STATIC = pathlib.Path(__file__).parent / "static"


class TranslateRequest(BaseModel):
    text: str
    api_key: str


class SolveRequest(BaseModel):
    """Loose on purpose: `dict` and `list`, not a precise shape.

    Real checking belongs to Puzzle._validate, clues_from_json and
    validate_constraints, which already exist and give complaints written for a
    person. Describing the shape twice would mean two boundaries to keep in
    step, and the weaker one would win by running first.
    """

    num_positions: int
    categories: dict
    clues: list


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/example")
def example():
    """The bundled puzzle, solved in one step. The no-key path, and the one
    that proves the solver works with no AI involved at all."""
    info, puzzle, clues = load()
    return {"info": info, **_solved(puzzle, clues)}


@app.post("/api/translate")
def translate_puzzle(request: TranslateRequest):
    """English in, clues out — and deliberately no answer yet."""
    if not request.api_key.strip():
        raise HTTPException(400, "An API key is needed to translate a new puzzle.")
    if not request.text.strip():
        raise HTTPException(400, "There is no puzzle here to read.")

    try:
        puzzle, clues = translate(request.text, anthropic_asker(request.api_key.strip()))
    except TranslationFailed as failed:
        # The clues could not be read even after retrying. Show what the guards
        # objected to — far more useful than "translation failed".
        raise HTTPException(422, {
            "message": "Could not read that puzzle into clues.",
            "attempts": failed.attempts,
        })
    except Exception as broken:  # noqa: BLE001 - the API call can fail many ways
        raise HTTPException(502, f"The translation service failed: {broken}")

    return {
        "info": {"name": "Your puzzle", "text": request.text, "question": ""},
        "num_positions": puzzle.num_positions,
        "categories": puzzle.categories,
        "clues": _clue_rows(clues),
    }


@app.post("/api/solve")
def solve_puzzle(request: SolveRequest):
    """Confirmed clues in, answer and explanation out.

    Every failure below means the data was wrong, not that the puzzle is
    impossible — so they are 400s the page can explain, never 500s.
    """
    try:
        puzzle = Puzzle(request.categories, request.num_positions)
    except (ValueError, TypeError, AttributeError) as bad:
        raise HTTPException(400, f"That is not a usable puzzle: {bad}")

    try:
        clues = clues_from_json(request.clues)
    except UnreadableClues as unreadable:
        raise HTTPException(400, {
            "message": "Those clues could not be read.",
            "problems": [f"{p.where}: {p.problem}" for p in unreadable.problems],
        })

    try:
        return _solved(puzzle, clues)
    except InvalidConstraint as invalid:
        # A clue naming something this puzzle never defined. Its own message
        # already lists every bad reference at once.
        raise HTTPException(400, {
            "message": "A clue named something this puzzle does not have.",
            "problems": [str(invalid)],
        })


def _clue_rows(clues):
    """Each clue three ways: the sentence it came from, what the solver
    actually understood, and the data itself so the browser can hand it back.

    `says` and `also_means` are built from the clue's fields and never from
    source_text. That is the whole difference between a check and a mirror —
    showing someone their own words back confirms nothing.
    """
    return [
        {"number": number,
         "source_text": clue.source_text,
         "says": clue.describe(),
         "also_means": clue.also_means(),
         "clue": clue_to_json(clue)}
        for number, clue in enumerate(clues, 1)
    ]


def _solved(puzzle, clues):
    """Run the solver and shape the result for the page.

    The trace is serialised with to_ai_payload — the same structure built for
    the AI wording layer. It already writes every fact out in full and gives
    each group a stable id, which is exactly what rendering needs too.
    """
    result = solve(puzzle, clues)

    return {
        "status": result.status.value,
        "reason": result.reason,
        # Names in order for the answer table; the values themselves so the
        # page can draw the grid of what was still possible at each step.
        "categories": list(puzzle.categories),
        "values": puzzle.categories,
        "num_positions": puzzle.num_positions,
        "answer": _answer_rows(puzzle, result.assignment),
        "clues": _clue_rows(clues),
        "trace": to_ai_payload(result.trace or [], clues),
    }


def _answer_rows(puzzle, assignment):
    """The finished grid as one row per position, or None if it never finished."""
    if assignment is None:
        return None
    return [
        {"position": position,
         "values": {category: assignment[(category, position)] for category in puzzle.categories}}
        for position in puzzle.positions
    ]
