import json

import pytest

from narrate import (
    NarrationFailed,
    for_writing,
    narrate,
    problems_with,
)


GROUPS = [
    {"id": 1, "kind": "eliminate", "says": ["the color at position 5 is not ivory"],
     "facts": [{"category": "color", "position": 5, "value": "ivory"}],
     "clue": {"kind": "clue", "number": 5, "text": "The green house is right of the ivory house."},
     "because": [], "children": [{"assuming": "x", "steps": []}]},
    {"id": 2, "kind": "conclude", "says": ["the color at position 1 is yellow"],
     "facts": [{"category": "color", "position": 1, "value": "yellow"}],
     "clue": {"kind": "rule", "text": "a value must be somewhere"},
     "because": [{"id": 1, "says": "the color at position 5 is not ivory"}],
     "children": []},
]


def replies(*canned):
    sent = []

    def ask(conversation):
        sent.append(conversation)
        return canned[len(sent) - 1]

    ask.sent = sent
    return ask


def prose_for(*sentences):
    return json.dumps({str(i): s for i, s in enumerate(sentences, 1)})


# ---------------------------------------------------------------------------
# What gets sent
# ---------------------------------------------------------------------------


def test_sub_proofs_and_raw_facts_are_not_sent():
    """Ids are numbered per level, so a flat reply could not address nested
    steps without collisions - and dropping them is what keeps this one call."""
    sent = for_writing(GROUPS)

    assert [set(step) for step in sent] == [{"id", "kind", "says", "clue", "because", "cases"}] * 2
    # the count survives; the sub-proofs themselves do not
    assert [step["cases"] for step in sent] == [1, 0]
    assert "children" not in json.dumps(sent)


def test_nothing_to_say_needs_no_call():
    ask = replies()  # would IndexError if called

    assert narrate([], ask) == {}


# ---------------------------------------------------------------------------
# The structural check - the only thing that CAN be checked about prose
# ---------------------------------------------------------------------------


def test_good_prose_comes_back_keyed_by_id():
    ask = replies(prose_for("House 5 isn't ivory.", "So house 1 must be yellow."))

    said = narrate(GROUPS, ask)

    assert said == {1: "House 5 isn't ivory.", 2: "So house 1 must be yellow."}
    assert len(ask.sent) == 1


def test_a_dropped_step_is_caught():
    """The wording cannot be checked, but which steps were addressed can. A
    step quietly skipped as "obvious" is exactly what this catches."""
    dropped = json.dumps({"1": "House 5 isn't ivory."})

    assert "left out" in " ".join(problems_with(json.loads(dropped), GROUPS))


def test_an_invented_step_is_caught():
    invented = {"1": "a", "2": "b", "3": "a step that does not exist"}

    assert "not steps in this proof" in " ".join(problems_with(invented, GROUPS))


def test_an_empty_sentence_is_caught():
    assert "empty sentence" in " ".join(problems_with({"1": "a", "2": "   "}, GROUPS))


def test_a_reply_that_is_not_an_object_is_caught():
    assert problems_with(["a", "b"], GROUPS)
    assert problems_with(None, GROUPS)


# ---------------------------------------------------------------------------
# Retry and give-up
# ---------------------------------------------------------------------------


def test_a_bad_reply_is_sent_back_with_what_was_wrong():
    ask = replies(json.dumps({"1": "only this one"}),
                  prose_for("House 5 isn't ivory.", "So house 1 must be yellow."))

    said = narrate(GROUPS, ask)

    assert len(said) == 2
    complaint = ask.sent[1][-1]["content"]
    assert "left out" in complaint and "2" in complaint


def test_giving_up_uses_none_of_the_prose():
    """Half-narrated is worse than not narrated: the reader could not tell
    which steps were skipped."""
    short = json.dumps({"1": "only this one"})

    with pytest.raises(NarrationFailed) as failed:
        narrate(GROUPS, replies(short, short))

    assert any("left out" in problem
               for attempt in failed.value.attempts for problem in attempt)


def test_markdown_fences_are_tolerated():
    fenced = "Sure:\n\n```json\n" + prose_for("one", "two") + "\n```"

    assert narrate(GROUPS, replies(fenced)) == {1: "one", 2: "two"}


def test_junk_is_a_problem_not_a_crash():
    with pytest.raises(NarrationFailed):
        narrate(GROUPS, replies("I'd rather not.", "Still no."))
