from deduction import StepKind
from examples import load
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
