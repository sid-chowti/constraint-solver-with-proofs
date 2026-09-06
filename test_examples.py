from deduction import StepKind
from examples import EXAMPLES, load, load_prose
from solve import Status, solve
from verify import verify


# The bundled example has to actually work, or the site's one no-key demo is
# broken and only a visitor would find out.
def test_the_einstein_example_solves():
    info, puzzle, clues = load()
    result = solve(puzzle, clues)

    assert result.status is Status.SOLVED
    assert verify(puzzle, clues, result.assignment) == []
    assert info["name"] == "The Einstein Puzzle"


# The two questions the puzzle actually asks.
def test_the_japanese_owns_the_zebra_and_the_norwegian_drinks_water():
    _, puzzle, clues = load()
    answer = solve(puzzle, clues).assignment

    house_with = lambda category, value: next(
        position for (c, position), v in answer.items() if c == category and v == value
    )

    assert answer[("nation", house_with("pet", "zebra"))] == "Japanese"
    assert answer[("nation", house_with("drink", "water"))] == "Norwegian"


# Every clue must carry the sentence it came from — that is what the
# explanation quotes back at the reader.
def test_every_clue_keeps_its_english_sentence():
    _, _, clues = load()

    assert len(clues) == 14
    assert all(clue.source_text for clue in clues)
    assert clues[0].source_text == "The Englishman lives in the red house."


# The displayed text must contain the clues themselves, since it is what a
# visitor reads and what they would re-translate with their own key.
def test_the_puzzle_text_contains_its_clues():
    info, _, clues = load()

    assert all(clue.source_text in info["text"] for clue in clues)


# The example is stored in the translator's own output format, so it doubles
# as a worked example of what the AI is asked to produce.
def test_the_example_produces_a_full_explanation():
    _, puzzle, clues = load()
    result = solve(puzzle, clues)

    assert len(result.trace) > 50
    assert any(step.children for group in result.trace for step in group.steps)
    assert any(group.kind is StepKind.CONCLUDE for group in result.trace)


# ---------------------------------------------------------------------------
# The stored wording for the bundled puzzle
# ---------------------------------------------------------------------------


# THE important one. The prose is written once and checked in, so any change to
# the solver, the grouping, or the trace can silently leave it describing steps
# that no longer exist - and a proof narrated with the wrong sentences is worse
# than one with none. This is the same structural check the AI's own replies
# have to pass.
def test_the_stored_prose_still_matches_the_trace():
    from deduction import to_ai_payload
    from examples import load, load_prose
    from narrate import for_writing, problems_with
    from solve import solve

    _, puzzle, clues = load()
    groups = for_writing(to_ai_payload(solve(puzzle, clues).trace, clues))
    prose = load_prose()

    assert prose is not None
    assert problems_with({str(k): v for k, v in prose.items()}, groups) == []


def test_every_stored_sentence_is_real_english():
    """Cheap sanity: the stored file is hand-written, so a truncated or
    placeholder entry would otherwise sit there unnoticed."""
    from examples import load_prose

    for step, sentence in load_prose().items():
        assert len(sentence) > 20, f"step {step} has a stub sentence"
        # A sentence may legitimately open with a quoted clue.
        assert sentence.lstrip('"“').lstrip()[0].isupper(), f"step {step} does not start a sentence"
        assert sentence.rstrip()[-1] in ".!?", f"step {step} is unfinished"


def test_a_puzzle_with_no_stored_prose_says_so():
    from examples import load_prose

    assert load_prose("nothing_is_stored_here") is None


# ---------------------------------------------------------------------------
# Every bundled puzzle, not just the famous one
# ---------------------------------------------------------------------------

import pytest

from examples import available

NAMES = [example["name"] for example in available()]


def test_there_is_more_than_one_puzzle_to_try():
    """Most visitors will never have an API key, so the bundled puzzles are the
    whole site for them."""
    assert len(NAMES) >= 3


@pytest.mark.parametrize("name", NAMES)
def test_every_bundled_puzzle_solves(name):
    from solve import Status, solve

    _, puzzle, clues = load(name)
    result = solve(puzzle, clues)

    assert result.status is Status.SOLVED, f"{name} does not solve"
    # SOLVED means the grid was fully forced and the answer re-verified, which
    # for a sound solver means this is the ONLY answer - so a bundled puzzle
    # can never be quietly ambiguous.
    assert result.trace


@pytest.mark.parametrize("name", NAMES)
def test_every_bundled_puzzle_names_its_values_in_its_own_text(name):
    """The same rule the grounding guard applies to the AI. A bundled puzzle
    that fails it would be a puzzle whose text does not mention something the
    reader is expected to place."""
    import json

    from grounding import find_ungrounded

    data = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))

    assert find_ungrounded(data["categories"], data["text"]) == []


@pytest.mark.parametrize("name", NAMES)
def test_no_bundled_puzzle_has_a_clue_it_does_not_need(name):
    """A minimal puzzle is a better demo: every clue in the list is doing work,
    which is what makes the step-by-step worth reading."""
    from solve import Status, solve

    _, puzzle, clues = load(name)

    for dropped in range(len(clues)):
        fewer = clues[:dropped] + clues[dropped + 1:]
        assert solve(puzzle, fewer).status is not Status.SOLVED, (
            f"{name} still solves without clue {dropped + 1}")


@pytest.mark.parametrize("name", NAMES)
def test_stored_prose_where_present_still_matches_the_trace(name):
    """Prose is written once and checked in, so it can silently drift when the
    solver or the grouping changes. Any example that ships wording gets the
    same structural check the AI's own replies must pass."""
    from deduction import to_ai_payload
    from narrate import for_writing, problems_with
    from solve import solve

    prose = load_prose(name)
    if prose is None:
        return  # this puzzle ships without wording, which is allowed

    _, puzzle, clues = load(name)
    groups = for_writing(to_ai_payload(solve(puzzle, clues).trace, clues))

    assert problems_with({str(k): v for k, v in prose.items()}, groups) == []
