import json

from grounding import complaints, find_ungrounded, is_grounded


RED_HOUSE = "The Englishman lives in the red house. The Spaniard owns the dog."


# The point of the guard: a value nobody wrote is a value the AI made up.
def test_an_invented_value_is_caught():
    assert is_grounded("dog", RED_HOUSE) is True
    assert is_grounded("unicorn", RED_HOUSE) is False


# ---------------------------------------------------------------------------
# Everything below is a way this guard would WRONGLY refuse a good puzzle.
# Each one is a real spelling from the bundled Einstein puzzle. A false alarm
# is worse than a miss here: a miss still reaches the confirmation screen where
# a person sees every value, but a false alarm stops them ever getting there.
# ---------------------------------------------------------------------------


def test_a_value_that_is_only_part_of_a_word_still_counts():
    """The puzzle says "Englishman"; the AI quite reasonably calls the nation
    "English"."""
    assert is_grounded("English", "The Englishman lives in the red house.") is True


def test_a_value_with_the_spaces_taken_out_still_counts():
    """The puzzle says "Old Gold" and "Lucky Strike"; the AI writes them as one
    word, because a value with a space in it is awkward to key on."""
    assert is_grounded("OldGold", "The Old Gold smoker owns snails.") is True
    assert is_grounded("LuckyStrike", "The Lucky Strike smoker drinks juice.") is True
    assert is_grounded("OldGold", "The Old-Gold smoker owns snails.") is True


def test_plurals_match_in_both_directions():
    assert is_grounded("Kools", "Kool is smoked in the yellow house.") is True
    assert is_grounded("Kool", "Kools are smoked in the yellow house.") is True


def test_matching_ignores_case():
    assert is_grounded("blue", "The Norwegian lives next to the BLUE house.") is True


# Permissive, but not so permissive it matches anything. A short invented value
# must not be found hiding inside a longer real word.
def test_a_value_hiding_inside_another_word_does_not_count():
    assert is_grounded("at", "The cat sat down.") is False
    assert is_grounded("cat", "The cat sat down.") is True


# The real acceptance test: the bundled puzzle is known-good and hand-checked,
# so every one of its values must ground. A guard that refuses this puzzle is
# broken, whatever else it gets right.
def test_the_bundled_puzzle_is_fully_grounded():
    data = json.loads(open("examples/einstein.json", encoding="utf-8").read())

    assert find_ungrounded(data["categories"], data["text"]) == []


def test_complaints_name_the_category_and_the_value():
    said = complaints({"pet": ["dog", "unicorn"]}, RED_HOUSE)

    assert len(said) == 1
    assert "unicorn" in said[0] and "pet" in said[0]
