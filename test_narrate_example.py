import json

import pytest

from narrate_example import main, regenerate


def canned_ask(conversation):
    """Stand in for the API: answer every id it was actually asked about."""
    asked = json.loads(conversation[0]["content"].split("\n\n", 1)[1])
    return json.dumps({str(step["id"]): f"Step {step['id']} said plainly."
                       for step in asked})


def lazy_ask(conversation):
    """A model that skips steps it thinks are obvious."""
    return json.dumps({"1": "Only the first one."})


# The happy path: real trace in, a file that matches it out.
def test_regenerating_writes_a_file_that_matches_the_trace(tmp_path):
    from deduction import to_ai_payload
    from examples import load
    from narrate import for_writing, problems_with
    from solve import solve

    path, count = regenerate("desks", canned_ask, out_dir=tmp_path)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "desks_prose.json"
    assert count == len(written["prose"])

    _, puzzle, clues = load("desks")
    groups = for_writing(to_ai_payload(solve(puzzle, clues).trace, clues))
    assert problems_with(written["prose"], groups) == []


# The note has to describe a command that exists - the previous one named a
# script that was never written.
def test_the_stored_note_names_a_real_command(tmp_path):
    path, _ = regenerate("desks", canned_ask, out_dir=tmp_path)

    note = json.loads(path.read_text(encoding="utf-8"))["_note"]
    assert "narrate_example" in note and "--name desks" in note


# Nothing is written unless it lines up. A half-narrated file would have the
# demo confidently describing steps that are not there.
def test_wording_that_does_not_line_up_is_never_written(tmp_path):
    from narrate import NarrationFailed

    # regenerate() lets the failure through; main() is what turns it into a
    # clean exit message for someone at a terminal.
    with pytest.raises(NarrationFailed):
        regenerate("desks", lazy_ask, out_dir=tmp_path)

    assert list(tmp_path.iterdir()) == [], "a rejected narration must leave no file"


def test_the_command_line_reports_giving_up_cleanly(tmp_path, monkeypatch):
    import narrate_example

    monkeypatch.setattr(narrate_example, "anthropic_asker", lambda key: lazy_ask)
    monkeypatch.setattr(narrate_example, "EXAMPLES", tmp_path)

    with pytest.raises(SystemExit) as raised:
        main(["--api-key", "k", "--name", "desks"])

    assert "gave up on desks" in str(raised.value)


def test_an_unknown_puzzle_is_refused():
    with pytest.raises(SystemExit) as raised:
        main(["--api-key", "k", "--name", "no-such-puzzle"])

    assert "no bundled puzzle" in str(raised.value)


def test_saying_nothing_at_all_is_refused():
    with pytest.raises(SystemExit) as raised:
        main(["--api-key", "k"])

    assert "--name" in str(raised.value)
