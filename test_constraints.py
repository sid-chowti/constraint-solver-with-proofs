import pytest

from constraints import (
    AbsolutePosition,
    And,
    Or,
    RelativePosition,
    BadReference,
    InvalidConstraint,
    validate_constraints,
)
from possibilities import Contradiction, PossibilityGrid
from puzzle import Puzzle


# Shared 4-position, single-category puzzle.
def make_puzzle():
    return Puzzle({"color": ["red", "green", "blue", "yellow"]}, 4)


def make_grid():
    return PossibilityGrid(make_puzzle())


# "==" pins the value to one spot and clears every other value from that spot.
def test_absolute_equals_pins_value_and_clears_the_position():
    grid = make_grid()

    changed = AbsolutePosition(("color", "green"), "==", 2).propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "green") == [2]
    assert grid.candidates("color", 2) == {"green"}


# "!=" removes exactly one candidate and leaves the rest alone.
def test_absolute_not_equals_removes_only_that_candidate():
    grid = make_grid()

    changed = AbsolutePosition(("color", "red"), "!=", 1).propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "red") == [2, 3, 4]
    # Nothing else at position 1 was touched.
    assert grid.candidates("color", 1) == {"green", "blue", "yellow"}


# "<" is strict: the value is ruled out at the named position and above.
def test_absolute_less_than_rules_out_that_position_and_above():
    grid = make_grid()

    changed = AbsolutePosition(("color", "red"), "<", 3).propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "red") == [1, 2]


def test_absolute_greater_than_rules_out_that_position_and_below():
    grid = make_grid()

    changed = AbsolutePosition(("color", "red"), ">", 2).propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "red") == [3, 4]


# Running the same clue twice must report "nothing changed" the second time,
# otherwise the solver loop would never reach a fixed point.
def test_absolute_propagate_is_idempotent():
    grid = make_grid()
    constraint = AbsolutePosition(("color", "green"), "==", 2)

    assert constraint.propagate(grid) is True
    assert constraint.propagate(grid) is False


# Bad operators are caught at construction, not deep inside propagation.
# ("<=" used to be the example here — it's a real operator now.)
def test_unsupported_operator_rejected_at_construction():
    with pytest.raises(ValueError, match="unsupported operator"):
        AbsolutePosition(("color", "red"), "=<", 2)

    with pytest.raises(ValueError, match="unsupported operator"):
        RelativePosition(("color", "red"), ("color", "green"), "~", 1)


# "Somewhere to the left": the leftmost spot has nothing to its left to pair
# with, and the rightmost has nothing to its right.
def test_relative_less_than_trims_both_ends():
    grid = make_grid()

    changed = RelativePosition(("color", "green"), ("color", "blue"), "<").propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "green") == [1, 2, 3]
    assert grid.positions_for("color", "blue") == [2, 3, 4]


def test_relative_greater_than_trims_both_ends():
    grid = make_grid()

    changed = RelativePosition(("color", "green"), ("color", "blue"), ">").propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "green") == [2, 3, 4]
    assert grid.positions_for("color", "blue") == [1, 2, 3]


# "Immediately left of" is offset 1 with "==": pos(green) + 1 == pos(blue).
def test_relative_immediately_left_trims_one_spot_from_each_end():
    grid = make_grid()

    changed = RelativePosition(("color", "green"), ("color", "blue"), "==", 1).propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "green") == [1, 2, 3]
    assert grid.positions_for("color", "blue") == [2, 3, 4]


# The user's worked example: green left of blue, blue already down to 1 or 2.
# Blue can't be 1 (nothing is left of house 1), so blue is 2 and green is 1.
def test_relative_narrows_when_the_partner_is_already_narrowed():
    puzzle = Puzzle({"color": ["red", "green", "blue"]}, 3)
    grid = PossibilityGrid(puzzle)
    grid.eliminate("color", 3, "blue")

    RelativePosition(("color", "green"), ("color", "blue"), "<").propagate(grid)

    assert grid.positions_for("color", "green") == [1]
    assert grid.positions_for("color", "blue") == [2]


# A pinned partner leaves only one illegal spot for "!=".
def test_relative_not_equals_only_bites_when_partner_is_pinned():
    grid = make_grid()
    for position in [1, 2, 4]:
        grid.eliminate("color", position, "blue")

    changed = RelativePosition(("color", "green"), ("color", "blue"), "!=").propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "green") == [1, 2, 4]


# Cross-category, offset 0: "the same position as".
def test_relative_equals_links_two_categories():
    puzzle = Puzzle({"person": ["anna", "ben"], "pet": ["cat", "dog"]}, 2)
    grid = PossibilityGrid(puzzle)
    grid.eliminate("person", 1, "anna")

    RelativePosition(("person", "anna"), ("pet", "cat"), "==").propagate(grid)

    assert grid.positions_for("pet", "cat") == [2]


# Once arc consistent, a second pass must report no change so the outer
# fixed-point loop can stop.
def test_relative_propagate_is_idempotent():
    grid = make_grid()
    constraint = RelativePosition(("color", "green"), ("color", "blue"), "==", 1)

    assert constraint.propagate(grid) is True
    assert constraint.propagate(grid) is False


# A negative offset flips the direction: pos(green) - 1 == pos(blue) means
# green sits immediately to the RIGHT of blue.
def test_relative_negative_offset_reverses_the_direction():
    grid = make_grid()

    changed = RelativePosition(("color", "green"), ("color", "blue"), "==", -1).propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "green") == [2, 3, 4]
    assert grid.positions_for("color", "blue") == [1, 2, 3]


# An offset bigger than the board leaves no legal pair at all, so both sides
# lose every position. propagate() itself doesn't raise here — no single spot
# runs out of values — the loop's "every value needs a home" rule catches it.
def test_relative_impossible_offset_empties_both_sides():
    grid = make_grid()

    RelativePosition(("color", "green"), ("color", "blue"), "==", 9).propagate(grid)

    assert grid.positions_for("color", "green") == []
    assert grid.positions_for("color", "blue") == []


# allows() is the yes/no pair test the sweep is built on.
def test_allows_encodes_operator_and_offset():
    immediately_left = RelativePosition(("color", "green"), ("color", "blue"), "==", 1)

    assert immediately_left.allows(1, 2) is True
    assert immediately_left.allows(2, 1) is False
    assert immediately_left.allows(1, 3) is False


# And runs every child, so eliminations from all of them land.
def test_and_propagates_all_children():
    grid = make_grid()

    changed = And([
        AbsolutePosition(("color", "red"), "!=", 1),
        AbsolutePosition(("color", "green"), "!=", 2),
    ]).propagate(grid)

    assert changed is True
    assert not grid.is_candidate("color", 1, "red")
    assert not grid.is_candidate("color", 2, "green")


def test_and_reports_no_change_once_children_are_settled():
    grid = make_grid()
    constraint = And([AbsolutePosition(("color", "red"), "!=", 1)])

    assert constraint.propagate(grid) is True
    assert constraint.propagate(grid) is False


# Mixed children: the absolute clue runs first and knocks green off house 1,
# and the relative clue immediately feeds on that narrower green inside the
# same pass — so blue loses house 2 as well as house 1.
def test_and_mixes_absolute_and_relative_children():
    grid = make_grid()

    changed = And([
        AbsolutePosition(("color", "green"), "!=", 1),
        RelativePosition(("color", "green"), ("color", "blue"), "==", 1),
    ]).propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "green") == [2, 3]
    assert grid.positions_for("color", "blue") == [3, 4]


# And recurses into nested And, so a bundled group prunes like a flat list.
def test_and_recurses_into_nested_and():
    grid = make_grid()

    And([
        And([
            AbsolutePosition(("color", "green"), "!=", 1),
            RelativePosition(("color", "green"), ("color", "blue"), "==", 1),
        ]),
        AbsolutePosition(("color", "blue"), "!=", 4),
    ]).propagate(grid)

    assert grid.positions_for("color", "green") == [2, 3]
    assert grid.positions_for("color", "blue") == [3]


# One branch means there's nothing to choose between, so Or acts like the
# branch itself.
def test_or_with_a_single_branch_applies_that_branch():
    grid = make_grid()

    changed = Or([AbsolutePosition(("color", "red"), "==", 1)]).propagate(grid)

    assert changed is True
    assert grid.positions_for("color", "red") == [1]


# Two live branches that rule out different things agree on nothing, so
# nothing may be cut. Branch 1 allows house 2, branch 2 allows house 1.
def test_or_eliminates_nothing_while_both_branches_survive():
    grid = make_grid()

    changed = Or([
        AbsolutePosition(("color", "red"), "!=", 1),
        AbsolutePosition(("color", "red"), "!=", 2),
    ]).propagate(grid)

    assert changed is False
    assert grid.positions_for("color", "red") == [1, 2, 3, 4]


# Both branches agree red isn't house 3 or 4, so that much is safe to cut
# even though which branch is true is still unknown.
def test_or_cuts_what_every_branch_agrees_on():
    grid = make_grid()

    Or([
        AbsolutePosition(("color", "red"), "==", 1),
        AbsolutePosition(("color", "red"), "==", 2),
    ]).propagate(grid)

    assert grid.positions_for("color", "red") == [1, 2]
    # Green survives everywhere: it is free in whichever branch red doesn't take.
    assert grid.positions_for("color", "green") == [1, 2, 3, 4]


# Killing a branch is the whole point. Red is pinned to house 2 up front, so
# the "red is house 1" branch dies and the other branch is applied for real.
def test_or_applies_the_only_branch_that_survives():
    grid = make_grid()
    AbsolutePosition(("color", "red"), "==", 2).propagate(grid)

    Or([
        AbsolutePosition(("color", "green"), "==", 2),
        AbsolutePosition(("color", "green"), "==", 3),
    ]).propagate(grid)

    assert grid.positions_for("color", "green") == [3]


# A branch that only dies once the PUZZLE RULES run, not from its own clue.
# Blue is squeezed into house 1 only. The "green is house 1" branch clears blue
# out of house 1, leaving blue homeless — which just eliminate() cannot see.
def test_or_kills_a_branch_using_the_puzzle_rules():
    puzzle = Puzzle({"color": ["red", "green", "blue"]}, 3)
    grid = PossibilityGrid(puzzle)
    grid.eliminate("color", 2, "blue")
    grid.eliminate("color", 3, "blue")

    Or([
        AbsolutePosition(("color", "green"), "==", 1),
        AbsolutePosition(("color", "green"), "==", 2),
    ]).propagate(grid)

    assert grid.positions_for("color", "green") == [2]


# Every branch impossible means the Or itself is impossible.
def test_or_raises_when_every_branch_dies():
    grid = make_grid()
    AbsolutePosition(("color", "red"), "==", 4).propagate(grid)

    with pytest.raises(Contradiction, match="no branch"):
        Or([
            AbsolutePosition(("color", "red"), "==", 1),
            AbsolutePosition(("color", "red"), "==", 2),
        ]).propagate(grid)


# An Or of nothing has no way to be true.
def test_or_with_no_branches_is_impossible():
    with pytest.raises(Contradiction, match="no branch"):
        Or([]).propagate(make_grid())


# The trial copies must not leak: a dead branch's eliminations are thrown away.
def test_or_does_not_leak_a_dead_branch_into_the_real_grid():
    grid = make_grid()
    AbsolutePosition(("color", "red"), "==", 1).propagate(grid)

    # Branch 1 (green in house 1) is impossible — red already owns house 1.
    Or([
        AbsolutePosition(("color", "green"), "==", 1),
        AbsolutePosition(("color", "green"), "==", 2),
    ]).propagate(grid)

    # If the dead branch had leaked, red would have been cleared from house 1.
    assert grid.positions_for("color", "red") == [1]
    assert grid.positions_for("color", "green") == [2]


# Once settled, a second pass must report no change so the outer loop can stop.
def test_or_propagate_is_idempotent():
    grid = make_grid()
    constraint = Or([
        AbsolutePosition(("color", "red"), "==", 1),
        AbsolutePosition(("color", "red"), "==", 2),
    ])

    assert constraint.propagate(grid) is True
    assert constraint.propagate(grid) is False


# "Next to" is the clue Or exists for: left of OR right of.
def test_or_expresses_next_to():
    puzzle = Puzzle({"color": ["red", "green", "blue"]}, 3)
    grid = PossibilityGrid(puzzle)
    AbsolutePosition(("color", "red"), "==", 1).propagate(grid)

    # Green is next to red.
    Or([
        RelativePosition(("color", "green"), ("color", "red"), "==", 1),
        RelativePosition(("color", "green"), ("color", "red"), "==", -1),
    ]).propagate(grid)

    # Red is house 1, so "green immediately right of red" is the only option.
    assert grid.positions_for("color", "green") == [2]


# validate_constraints guards the boundary where AI-written constraints arrive.
def test_validate_accepts_a_well_formed_constraint():
    validate_constraints(make_puzzle(), [AbsolutePosition(("color", "red"), "==", 1)])


def test_validate_rejects_unknown_category():
    with pytest.raises(ValueError, match="not a category"):
        validate_constraints(make_puzzle(), [AbsolutePosition(("drink", "tea"), "==", 1)])


def test_validate_rejects_unknown_value():
    with pytest.raises(ValueError, match="not a valid value"):
        validate_constraints(make_puzzle(), [AbsolutePosition(("color", "purple"), "==", 1)])


# Both sides of a relative clue get checked.
def test_validate_checks_both_ends_of_relative_position():
    constraint = RelativePosition(("color", "red"), ("color", "purple"), "==", 1)

    with pytest.raises(ValueError, match="not a valid value"):
        validate_constraints(make_puzzle(), [constraint])


def test_validate_recurses_into_combinators():
    nested = And([Or([AbsolutePosition(("color", "purple"), "==", 1)])])

    with pytest.raises(ValueError, match="not a valid value"):
        validate_constraints(make_puzzle(), [nested])


def test_validate_rejects_unknown_constraint_type():
    with pytest.raises(TypeError, match="unknown constraint type"):
        validate_constraints(make_puzzle(), [object()])


# --- InvalidConstraint: the AI trust boundary ------------------------------


# Still a ValueError underneath, so older `except ValueError` keeps working.
def test_invalid_constraint_is_still_a_value_error():
    assert issubclass(InvalidConstraint, ValueError)


# The point of the whole thing: pieces a retry can read, not a sentence to parse.
def test_invalid_constraint_carries_structured_pieces():
    clue = AbsolutePosition(("color", "purple"), "==", 1)

    with pytest.raises(InvalidConstraint) as caught:
        validate_constraints(make_puzzle(), [clue])

    problem = caught.value.problems[0]
    assert problem == BadReference(
        clue=clue, kind="value", category="color", value="purple",
        allowed=["red", "green", "blue", "yellow"],
    )


# An unknown CATEGORY reports the category names as the alternatives.
def test_unknown_category_offers_the_known_categories():
    with pytest.raises(InvalidConstraint) as caught:
        validate_constraints(make_puzzle(), [AbsolutePosition(("drink", "tea"), "==", 1)])

    problem = caught.value.problems[0]
    assert problem.kind == "category"
    assert "color" in problem.allowed


# Every mistake at once, so the AI gets one retry instead of one per mistake.
def test_all_problems_are_reported_together():
    clues = [
        AbsolutePosition(("color", "purple"), "==", 1),
        RelativePosition(("color", "orange"), ("drink", "tea"), "<", 0),
    ]

    with pytest.raises(InvalidConstraint) as caught:
        validate_constraints(make_puzzle(), clues)

    assert [p.value for p in caught.value.problems] == ["purple", "orange", "tea"]


# Clean clues raise nothing, even in bulk.
def test_validate_accepts_a_list_of_good_constraints():
    validate_constraints(make_puzzle(), [
        AbsolutePosition(("color", "red"), "==", 1),
        RelativePosition(("color", "green"), ("color", "blue"), "<", 0),
    ])


# A clue naming house 47 in a five-house puzzle is a BROKEN CLUE, not an
# impossible puzzle. Without this, propagation crosses the value off
# everywhere and reports UNSOLVABLE — telling a user their good puzzle is
# broken, when the AI is the thing that needs retrying.
def test_a_position_outside_the_puzzle_is_rejected():
    puzzle = Puzzle({"color": ["red", "green", "blue"]}, 3)

    with pytest.raises(InvalidConstraint) as raised:
        validate_constraints(puzzle, [AbsolutePosition(("color", "red"), "==", 47)])

    problem = raised.value.problems[0]
    assert problem.kind == "position"
    assert problem.value == 47
    assert problem.allowed == [1, 2, 3]


# Position zero is outside a 1-indexed puzzle too — an easy off-by-one for a
# translator that thinks in zero-based arrays.
def test_position_zero_is_rejected():
    puzzle = Puzzle({"color": ["red", "green", "blue"]}, 3)

    with pytest.raises(InvalidConstraint):
        validate_constraints(puzzle, [AbsolutePosition(("color", "red"), "==", 0)])


# A bad position nested inside a combined clue is still found.
def test_a_bad_position_inside_a_group_is_found():
    puzzle = Puzzle({"color": ["red", "green", "blue"]}, 3)
    clue = Or([AbsolutePosition(("color", "red"), "==", 1),
               AbsolutePosition(("color", "red"), "==", 9)])

    with pytest.raises(InvalidConstraint) as raised:
        validate_constraints(puzzle, [clue])

    assert any(p.kind == "position" for p in raised.value.problems)


# ---------------------------------------------------------------------------
# describe(): what the confirmation screen shows the user
# ---------------------------------------------------------------------------

GREEN = ("color", "green")
IVORY = ("color", "ivory")
DOG = ("pet", "dog")


def test_mirrored_agrees_with_allows_on_every_shape():
    """The mirror rule, checked against the propagation code that already
    works: for every operator and offset, mirrored().allows(q, p) must agree
    with allows(p, q) on every pair of positions. This is what lets describe()
    build the second reading from the structure instead of from the words."""
    for operator in ("==", "!=", "<", ">", "<=", ">="):
        for offset in range(-3, 4):
            clue = RelativePosition(GREEN, IVORY, operator, offset)
            mirror = clue.mirrored()
            for a in range(1, 8):
                for b in range(1, 8):
                    assert clue.allows(a, b) == mirror.allows(b, a), (operator, offset, a, b)


def test_somewhere_never_becomes_immediately():
    """The bug this whole design guards against: "somewhere right of" and
    "immediately right of" are different clues, and mirroring must not turn one
    into the other. Concretely, green in house 1 with ivory in house 4
    satisfies the loose clue but not the adjacent one."""
    somewhere = RelativePosition(IVORY, GREEN, ">", 0)   # ivory anywhere right of green
    adjacent = RelativePosition(IVORY, GREEN, "==", 1)   # ivory exactly one left of green

    assert somewhere.allows(4, 1) is True
    assert adjacent.allows(4, 1) is False

    # The wording keeps them apart, in both readings.
    for text in (somewhere.describe(), somewhere.also_means()):
        assert "somewhere" in text
        assert "immediately" not in text
    for text in (adjacent.describe(), adjacent.also_means()):
        assert "immediately" in text
        assert "somewhere" not in text

    # And the mirror keeps the offset that carries the strength.
    assert somewhere.mirrored().offset == 0
    assert adjacent.mirrored().offset == -1


def test_direction_clues_are_said_both_ways():
    """The reader must never have to flip a direction in their head."""
    clue = RelativePosition(IVORY, GREEN, "==", 1)
    assert clue.describe() == "ivory (color) is immediately left of green (color)"
    assert clue.also_means() == "green (color) is immediately right of ivory (color)"


def test_symmetric_clues_have_no_second_reading():
    """"A and B share a house" says the same thing whichever side leads, so a
    second sentence would be noise."""
    assert RelativePosition(GREEN, DOG, "==", 0).also_means() is None
    assert RelativePosition(GREEN, DOG, "!=", 0).also_means() is None
    assert AbsolutePosition(GREEN, "==", 1).also_means() is None


def test_describe_never_echoes_source_text():
    """describe() must read the fields, not the sentence. If it ever fell back
    to source_text the screen would show the user their own words and confirm
    nothing at all."""
    lie = RelativePosition(IVORY, GREEN, "==", 1,
                           source_text="The green house is left of the ivory house.")
    assert "The green house is left" not in lie.describe()
    assert lie.describe() == "ivory (color) is immediately left of green (color)"


def test_next_to_is_recognised():
    """Every "next to" clue arrives as a two-branch Or; say it the way the
    puzzle said it."""
    next_to = Or([RelativePosition(GREEN, DOG, "==", 1),
                  RelativePosition(GREEN, DOG, "==", -1)])
    assert next_to.describe() == "green (color) is next to dog (pet)"

    two_apart = Or([RelativePosition(GREEN, DOG, "==", 2),
                    RelativePosition(GREEN, DOG, "==", -2)])
    assert two_apart.describe() == "green (color) is 2 houses away from dog (pet)"


def test_an_unrelated_or_is_spelled_out():
    """Anything that isn't the neighbour shape gets listed branch by branch."""
    either = Or([AbsolutePosition(GREEN, "==", 1), AbsolutePosition(GREEN, "==", 5)])
    assert either.describe() == "either green (color) is in house 1, or green (color) is in house 5"


def test_unplanned_shape_falls_back_to_literal_wording():
    """No friendly guess for a combination nobody planned for - clunky and
    true beats readable and wrong."""
    odd = RelativePosition(GREEN, IVORY, ">", 2)
    assert odd.describe() == "the house of green (color) plus 2 is > the house of ivory (color)"


def test_absolute_wording_covers_every_operator():
    for operator in ("==", "!=", "<", ">", "<=", ">="):
        text = AbsolutePosition(GREEN, operator, 3).describe()
        assert "green (color)" in text and "house 3" in text


def test_and_joins_its_children():
    both = And([AbsolutePosition(GREEN, "==", 1), AbsolutePosition(DOG, "==", 2)])
    assert both.describe() == "green (color) is in house 1, and dog (pet) is in house 2"
