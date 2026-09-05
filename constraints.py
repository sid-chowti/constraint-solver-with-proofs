from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import operator as op

from deduction import SubProof, relevant
from possibilities import Contradiction
from rules import apply_all_rules


# Maps the operator strings constraints use onto Python's comparison functions.
_OPERATORS = {
    "==": op.eq,
    "!=": op.ne,
    "<": op.lt,
    ">": op.gt,
    "<=": op.le,
    ">=": op.ge,
}

# The mirror of each comparison, for saying a clue the other way round. Only
# the ORDER of the two sides changes, so "<" becomes ">" and the rest follow.
_REVERSED = {"==": "==", "!=": "!=", "<": ">", ">": "<", "<=": ">=", ">=": "<="}


def _pair_words(category_value):
    """A (category, value) pair as readable words, e.g. 'green (color)'."""
    category, value = category_value
    return f"{value} ({category})"


def _validate_operator(operator):
    """Raise ValueError if `operator` isn't one of the supported comparisons."""
    if operator not in _OPERATORS:
        raise ValueError(f"unsupported operator '{operator}', expected one of {sorted(_OPERATORS)}")


def _has_legal_partner(position, partner_positions, pair_is_legal):
    """True if at least one of `partner_positions` forms a legal pair with
    `position`. One surviving partner is enough to keep the position alive."""
    for partner_position in partner_positions:
        if pair_is_legal(position, partner_position):
            return True
    return False


class Constraint(ABC):
    """A clue. Knows how to rule candidates out of a PossibilityGrid."""

    @abstractmethod
    def propagate(self, possibilities):
        """Rule out candidates this clue proves impossible.
        Returns True if anything was actually eliminated."""

    def is_speculative(self):
        """True if working this clue out needs guessing at branches and
        copying the grid. Only Or does; everything else reads straight off
        the grid. The solver runs the cheap clues first."""
        return False

    @abstractmethod
    def describe(self):
        """This clue as one plain sentence, built ONLY from the clue's own
        fields.

        Deliberately never falls back to source_text. The confirmation screen
        shows this next to the user's sentence so they can spot a misreading;
        echoing their own words back would make it a mirror, not a check.
        """

    def also_means(self):
        """The same clue said with the two sides swapped, or None when that
        adds nothing. Only clues carrying a direction have a useful second
        reading."""
        return None


@dataclass
class AbsolutePosition(Constraint):
    """(category, value) compared to a literal position, e.g. Norwegian in house 1."""

    category_value: tuple
    operator: str
    position: int

    # The user's original English sentence, kept from translation time so the
    # explanation can quote their own words back instead of showing internals.
    # compare=False keeps it out of __eq__, so adding it changes no existing
    # behaviour — two clues that mean the same thing stay equal.
    source_text: str | None = field(default=None, compare=False)

    def __post_init__(self):
        _validate_operator(self.operator)

    def propagate(self, possibilities):
        category, value = self.category_value
        puzzle = possibilities.puzzle
        compare = _OPERATORS[self.operator]
        changed = False

        # Every operator does at least this: cross the value off each position
        # where the comparison comes out false. That alone fully handles
        # "!=", "<", ">", "<=" and ">=".
        # Nothing to lean on: this reads no grid state, only the clue and
        # arithmetic. These are the roots every other deduction traces back to.
        for position in puzzle.positions:
            if not compare(position, self.position):
                if possibilities.eliminate(category, position, value, because=self):
                    changed = True

        # "==" is the only one that pins both directions. The loop above
        # already said "this value lives nowhere else"; this adds the mirror
        # fact, that nothing else in the category can live here.
        if self.operator == "==":
            for other_value in puzzle.categories[category]:
                if other_value == value:
                    continue
                if possibilities.eliminate(category, self.position, other_value, because=self):
                    changed = True

        return changed

    # One wording per comparison against a literal house number.
    _WORDING = {
        "==": "{who} is in house {n}",
        "!=": "{who} is not in house {n}",
        "<": "{who} is somewhere left of house {n}",
        ">": "{who} is somewhere right of house {n}",
        "<=": "{who} is in house {n} or somewhere left of it",
        ">=": "{who} is in house {n} or somewhere right of it",
    }

    def describe(self):
        return self._WORDING[self.operator].format(
            who=_pair_words(self.category_value), n=self.position)


@dataclass
class RelativePosition(Constraint):
    """position(a) + offset {operator} position(b), e.g. green immediately left of white."""

    a: tuple
    b: tuple
    operator: str
    offset: int = 0

    source_text: str | None = field(default=None, compare=False)

    def __post_init__(self):
        _validate_operator(self.operator)

    def allows(self, position_a, position_b):
        """True if this pair of positions would satisfy the clue."""
        # Two DIFFERENT values of the SAME category can never share a position
        # — one house has one colour. Comparing numbers can't see that, so say
        # it outright. (Same category AND same value is just a clue about
        # itself, and sharing a position is fine there.)
        same_category = self.a[0] == self.b[0]
        different_values = self.a[1] != self.b[1]
        if same_category and different_values and position_a == position_b:
            return False

        compare = _OPERATORS[self.operator]
        return compare(position_a + self.offset, position_b)

    def mirrored(self):
        """The same clue with the two sides swapped: swap a and b, negate the
        offset, reverse the operator.

        This is how the second reading is produced — by mirroring the STRUCTURE
        and describing that, never by rewriting the first sentence's words.
        Swapping the words "left" and "right" in a string would eventually turn
        "somewhere right of" into "immediately left of", which invents a
        constraint the user never wrote.
        """
        return RelativePosition(self.b, self.a, _REVERSED[self.operator], -self.offset)

    # Wording for the shapes real puzzles use, keyed by (operator, offset).
    # The OFFSET is what carries the strength: "==" with offset +-1 means
    # adjacent, while "<" or ">" with offset 0 means any distance. mirrored()
    # preserves both numbers, so no mirror can cross between the two.
    _WORDING = {
        ("==", 0): "{a} and {b} are in the same house",
        ("==", 1): "{a} is immediately left of {b}",
        ("==", -1): "{a} is immediately right of {b}",
        ("<", 0): "{a} is somewhere left of {b}",
        (">", 0): "{a} is somewhere right of {b}",
    }

    def describe(self):
        wording = self._WORDING.get((self.operator, self.offset))
        if wording is not None:
            return wording.format(a=_pair_words(self.a), b=_pair_words(self.b))

        # A shape nothing planned for. Say it literally rather than reach for
        # friendly words: clunky and true beats readable and wrong.
        return (f"the house of {_pair_words(self.a)} plus {self.offset} "
                f"is {self.operator} the house of {_pair_words(self.b)}")

    def also_means(self):
        """Say direction clues both ways round, so the reader never has to flip
        "a is left of b" into "b is right of a" in their head. That flip is the
        exact mistake this screen exists to catch, so the screen must not ask
        the reader to perform it."""
        # Symmetric shapes: naming the other side first says nothing new.
        if self.offset == 0 and self.operator in ("==", "!="):
            return None
        return self.mirrored().describe()

    def propagate(self, possibilities):
        """Arc consistency, both directions: a position survives only if the
        other side has at least one surviving position it can pair with."""
        changed = False

        # Prune `a` first, then prune `b` against the already-narrowed `a`,
        # so a single pass squeezes out as much as it can.
        if self._prune(possibilities, self.a, self.b, self.allows):
            changed = True
        if self._prune(possibilities, self.b, self.a, lambda pb, pa: self.allows(pa, pb)):
            changed = True

        return changed

    def _prune(self, possibilities, target, partner, pair_is_legal):
        """Eliminate every position for `target` that no remaining position of
        `partner` can legally pair with. `pair_is_legal` takes the positions in
        (target, partner) order. Returns True if anything was eliminated."""
        target_category, target_value = target
        partner_category, partner_value = partner

        # Snapshot both sides up front — positions_for returns a fresh list,
        # so eliminating below can't disturb what we're looping over.
        partner_positions = possibilities.positions_for(partner_category, partner_value)
        target_positions = possibilities.positions_for(target_category, target_value)

        changed = False
        for target_position in target_positions:
            if _has_legal_partner(target_position, partner_positions, pair_is_legal):
                continue

            # This candidate died because every partner position that could
            # have paired with it is already gone. Those deaths are the proof.
            # If NO position was ever legal (say "immediately left of" tested
            # at the last house) there is nothing to cite and this is a root,
            # exactly like an AbsolutePosition.
            evidence = []
            for partner_position in possibilities.puzzle.positions:
                if not pair_is_legal(target_position, partner_position):
                    continue
                killer = possibilities.killed_by(partner_category, partner_position,
                                                 partner_value)
                if killer is not None:
                    evidence.append(killer)

            if possibilities.eliminate(target_category, target_position, target_value,
                                       because=self, leaning_on=evidence):
                changed = True

        return changed


@dataclass
class And(Constraint):
    """Holds only if every child holds, so every child may eliminate freely."""

    constraints: list

    source_text: str | None = field(default=None, compare=False)

    def propagate(self, possibilities):
        # No short-circuiting — each child may rule out different candidates.
        changed = False
        for constraint in self.constraints:
            if constraint.propagate(possibilities):
                changed = True
        return changed

    def is_speculative(self):
        """An And is only as cheap as its most expensive child."""
        return any(child.is_speculative() for child in self.constraints)

    def describe(self):
        return ", and ".join(child.describe() for child in self.constraints)


def _settle_branch(possibilities, constraint):
    """Push one branch as far as it goes on its own throwaway grid: the
    branch's clue plus both puzzle rules, over and over until nothing more
    falls out. Raises Contradiction if the branch turns out to be impossible."""
    changed = True
    while changed:
        changed = constraint.propagate(possibilities)
        if apply_all_rules(possibilities):
            changed = True


@dataclass
class Or(Constraint):
    """Holds if at least one child holds."""

    constraints: list

    source_text: str | None = field(default=None, compare=False)

    def propagate(self, possibilities):
        """Try every branch on its own copy. A value is only impossible if
        every surviving branch says so, so anything at least one branch still
        allows has to stay.

        Every branch keeps a sub-proof, survivors included. That is the part
        that differs from shaving: there, a surviving trial proves nothing and
        is thrown away, but here "white died under A, and white died under B"
        is exactly the argument, so a survivor's chain is load-bearing.
        """
        survivors = []
        branches = []  # one SubProof per branch, in the order the clue lists them

        for child in self.constraints:
            trial = possibilities.copy()
            try:
                _settle_branch(trial, child)
            except Contradiction as dead_end:
                # Trim to the chain that actually reached the contradiction —
                # a branch does plenty of work that never bore on it.
                branches.append(SubProof(
                    steps=relevant(trial.steps, [dead_end.step]),
                    refuted=True,
                    about=child,
                ))
                continue
            survivors.append((child, trial))

        # Every branch died, so the Or itself can't hold.
        if not survivors:
            possibilities.record_contradiction(
                "no branch of this Or can hold",
                because=self,
                children=branches,
            )

        return self._keep_union(possibilities, survivors, branches)

    def is_speculative(self):
        """Or is the one clue that has to guess and copy the grid."""
        return True

    def _neighbour_gap(self):
        """If this Or is the "N houses apart" shape - the same two things
        compared at +N and -N - return N. Otherwise None.

        Worth spotting because "next to" is how a puzzle says it in English,
        and every "next to" clue arrives as exactly this two-branch Or.
        """
        if len(self.constraints) != 2:
            return None
        left, right = self.constraints
        if not isinstance(left, RelativePosition) or not isinstance(right, RelativePosition):
            return None
        if (left.a, left.b) != (right.a, right.b):
            return None
        if left.operator != "==" or right.operator != "==":
            return None
        if left.offset != -right.offset or left.offset == 0:
            return None
        return abs(left.offset)

    def describe(self):
        gap = self._neighbour_gap()
        if gap is not None:
            a, b = _pair_words(self.constraints[0].a), _pair_words(self.constraints[0].b)
            return f"{a} is next to {b}" if gap == 1 else f"{a} is {gap} houses away from {b}"
        return "either " + ", or ".join(child.describe() for child in self.constraints)

    def _keep_union(self, possibilities, survivors, refutations):
        """Cross off values that NO surviving branch still allows. A value even
        one branch allows must stay — that branch could be the true one. With a
        single survivor this copies its eliminations across, so 'only one
        branch left' needs no special case."""
        puzzle = possibilities.puzzle
        changed = False

        for category in puzzle.categories:
            for position in puzzle.positions:
                # candidates() hands back a copy, so eliminating below can't
                # disturb what we're looping over.
                for value in possibilities.candidates(category, position):
                    if any(t.is_candidate(category, position, value) for _, t in survivors):
                        continue
                    if not possibilities.eliminate(category, position, value, because=self):
                        continue
                    changed = True

                    # The justification is every branch at once: the ones that
                    # died, plus the chain by which each survivor also killed
                    # this value. Look the step back up to hang them on it.
                    step = possibilities.killed_by(category, position, value)
                    step.children = list(refutations) + [
                        SubProof(
                            steps=relevant(trial.steps,
                                           [trial.killed_by(category, position, value)]),
                            refuted=False,
                            about=child,
                        )
                        for child, trial in survivors
                        if trial.killed_by(category, position, value) is not None
                    ]

        return changed


@dataclass
class BadReference:
    """One thing a clue named that the puzzle never defined. Deliberately
    pieces, not a sentence: an automatic retry needs to read `allowed` and feed
    it back to the AI, and digging that out of prose would be miserable."""

    clue: object   # the constraint that named it
    kind: str      # "category", "value", or "position" — what was wrong
    category: str  # the category name the clue used
    value: str     # the value the clue used
    allowed: list  # what it could have said instead


class InvalidConstraint(ValueError):
    """Clues named things this puzzle never defined — the AI translation layer
    got something wrong. NOT the same as "this puzzle has no solution": here
    the solver never saw a real puzzle, so it knows nothing about whether an
    answer exists. Tell the AI to try again; don't tell the user their puzzle
    is broken.

    Subclasses ValueError so anything already catching ValueError still works.
    `problems` lists EVERY bad reference, so one trip back to the AI can fix
    them all instead of one trip per mistake."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__("; ".join(_describe(problem) for problem in problems))


def _describe(problem):
    """One bad reference as a readable sentence, for humans and logs. The
    structured fields on BadReference are what code should read."""
    if problem.kind == "position":
        return f"position {problem.value} is outside this puzzle (have: {problem.allowed})"
    if problem.kind == "category":
        return f"'{problem.category}' is not a category in this puzzle (have: {problem.allowed})"
    return (
        f"'{problem.value}' is not a valid value for category "
        f"'{problem.category}' (have: {problem.allowed})"
    )


def validate_constraints(puzzle, constraints):
    """The guard on the boundary where AI-written clues arrive. Plain Python,
    no AI: checks every (category, value) the clues mention against what the
    puzzle actually defines. Raises InvalidConstraint listing every problem
    found, or returns quietly if the clues are clean."""
    problems = []
    for constraint in constraints:
        _collect_bad_references(puzzle, constraint, problems)

    if problems:
        raise InvalidConstraint(problems)


def _collect_bad_references(puzzle, constraint, problems):
    """Walk one clue (and any children) adding every bad reference to
    `problems`. Collects rather than raising, so one pass finds them all."""
    if isinstance(constraint, AbsolutePosition):
        _check_reference(puzzle, constraint, constraint.category_value, problems)
        _check_position(puzzle, constraint, constraint.position, problems)
    elif isinstance(constraint, RelativePosition):
        _check_reference(puzzle, constraint, constraint.a, problems)
        _check_reference(puzzle, constraint, constraint.b, problems)
    elif isinstance(constraint, (And, Or)):
        for child in constraint.constraints:
            _collect_bad_references(puzzle, child, problems)
    else:
        # Not an AI mistake — something handed us an object that isn't a clue
        # at all. That's a bug in the calling code, so fail immediately.
        raise TypeError(f"unknown constraint type: {type(constraint).__name__}")


def _check_position(puzzle, clue, position, problems):
    """Record a BadReference if the clue names a position the puzzle has no
    room for — "house 47" in a five-house puzzle.

    Worth catching here rather than letting it through: propagation would
    cross the value off everywhere and report the PUZZLE as unsolvable, when
    really the CLUE is broken. Those two need different responses — one is
    shown to the user, the other is retried with the AI.
    """
    if position not in puzzle.positions:
        problems.append(BadReference(
            clue=clue, kind="position", category=None, value=position,
            allowed=list(puzzle.positions),
        ))


def _check_reference(puzzle, clue, category_value, problems):
    """Record a BadReference if (category, value) isn't real in this puzzle."""
    category, value = category_value

    if category not in puzzle.categories:
        problems.append(BadReference(
            clue=clue, kind="category", category=category, value=value,
            allowed=list(puzzle.categories),
        ))
        return  # No point checking the value against a category that doesn't exist.

    if value not in puzzle.categories[category]:
        problems.append(BadReference(
            clue=clue, kind="value", category=category, value=value,
            allowed=list(puzzle.categories[category]),
        ))
