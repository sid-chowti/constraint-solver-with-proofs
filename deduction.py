"""Recorded deduction steps, and the tools for turning them into something a
person can read.

Sits at the very bottom of the import stack — it imports nothing from this
project, so possibilities.py can use it without any risk of a circular import.
Same reasoning that put the puzzle rules in rules.py.

A Step is deliberately STRUCTURED, never English. The solver records what it
did and why; turning that into readable sentences is the AI layer's job later.
Keeping the split here is what lets the logic stay guaranteed-correct while the
wording stays flexible.
"""
from dataclasses import dataclass, field
from enum import Enum


class StepKind(Enum):
    """What a step is claiming. An Enum rather than strings so a typo is an
    instant crash instead of a condition that quietly never matches."""

    ELIMINATE = "eliminate"          # a candidate was ruled out
    CONCLUDE = "conclude"            # a cell came down to one value
    SUPPOSE = "suppose"              # an assumption, NOT a deduction
    CONTRADICTION = "contradiction"  # the puzzle broke; ends a sub-proof


class Rule(Enum):
    """The two puzzle rules, named so a step can say which one fired. Clues
    record themselves as `because`; the rules aren't objects, so they need
    these."""

    VALUE_USED_ONCE = "each value fills exactly one position"
    VALUE_MUST_BE_SOMEWHERE = "each value fills some position"
    SHAVING = "assuming it broke the puzzle"
    ASSUMPTION = "assumed, to see what follows"


@dataclass
class Step:
    """One thing the solver did, and its justification.

    Two separate structures live in this one record, and they answer different
    questions:

    - `children` is the NESTING (a tree). It holds the sub-proof from inside a
      pretend world — the chain a shaving refutation or an Or branch went
      through. It exists so a reader is never asked to take a refutation on
      faith, and so facts that were only true inside a pretence can't be
      mistaken for real ones.

    - `leaning_on` is the DEPENDENCY (a graph, not a tree — one step can feed
      many later steps and lean on many earlier ones). It is what makes a step
      say "because of step 12", and what a backwards walk follows to find the
      steps that actually mattered.
    """

    kind: StepKind

    # All three are optional because a CONTRADICTION may name only part of a
    # cell, or none of it: "nothing left
    # for color at house 3" has no value, "red has nowhere to go" has no
    # position. Every other kind sets both.
    #   "nothing left for color at house 3" has no value
    #   "red has nowhere to go"              has no position
    #   "no branch of this clue can hold"    has neither, nor a category
    # Every other kind fills in all three.
    category: str | None = None
    position: int | None = None
    value: str | None = None

    because: object = None  # the clue, or one of the Rule members above

    # default_factory, not [] — a bare [] as a default is shared by every
    # instance, so all Steps would end up appending into the same list.
    leaning_on: list = field(default_factory=list)
    children: list = field(default_factory=list)


def describe(step):
    """A short, factual label for one step. Deliberately plain and mechanical —
    this is NOT the explanation. It exists so the AI layer can be handed a step
    without having to look anything up; the readable prose is written later,
    from these."""
    where = f"position {step.position}"

    if step.kind is StepKind.ELIMINATE:
        return f"the {step.category} at {where} is not {step.value}"
    if step.kind is StepKind.CONCLUDE:
        return f"the {step.category} at {where} is {step.value}"
    if step.kind is StepKind.SUPPOSE:
        return f"suppose the {step.category} at {where} is {step.value}"

    # A contradiction names whichever part of the cell ran out, or none of it.
    if step.category is None:
        return "no branch of this clue can hold"
    if step.position is None:
        return f"no position is left for {step.value} in {step.category}"
    return f"no value is left for {step.category} at {where}"


@dataclass
class SubProof:
    """One pretend world, and how it turned out.

    Shaving produces exactly one: an assumption that ended in a contradiction.
    Or produces one per branch — and unlike shaving, its SURVIVING branches
    matter too. To justify "white is not in house 3" from "A or B", you have to
    show white died under A *and* under B; a survivor that agrees is part of
    the proof, not spare work.
    """

    steps: list
    refuted: bool          # True = ended in a contradiction
    about: object = None   # the branch clue, or the assumption being tested


def relevant(steps, targets):
    """The steps `targets` actually needed, in the order they happened.

    This is the backwards walk: start at the targets, follow `leaning_on`
    arrows backwards, and keep whatever you can reach. Everything else was
    true but beside the point.

    Plain graph reachability on the reversed dependency edges. `seen` guards
    against walking the same step twice, since one step can feed many others.
    """
    seen = set()
    stack = list(targets)

    while stack:
        step = stack.pop()
        if id(step) in seen:
            continue
        seen.add(id(step))
        stack.extend(step.leaning_on)

    # Filter the original list rather than returning `seen` directly, so the
    # result stays in the order things actually happened.
    return [step for step in steps if id(step) in seen]


@dataclass
class Group:
    """Several steps that share one reason, so they read as a single sentence.

    "Remove red from houses 2, 3, 4 and 5" is four steps but one thought:
    "since house 1 is red, no other house can be red."
    """

    because: object
    steps: list
    leaning_on: list = field(default_factory=list)

    @property
    def kind(self):
        """A group is always made of one kind of step, so the first one speaks
        for all of them."""
        return self.steps[0].kind


def group_steps(steps):
    """Merge neighbouring steps that share both a reason and the evidence they
    leaned on, so the trace reads in thoughts rather than in twitches.

    Only ELIMINATE steps merge. A conclusion, an assumption and a contradiction
    are each a single distinct thought, so they always stand alone.
    """
    groups = []

    for step in steps:
        merged = (
            groups
            and step.kind is StepKind.ELIMINATE
            and groups[-1].kind is StepKind.ELIMINATE
            and groups[-1].because is step.because
            # Same reason AND same evidence. Two eliminations from the same
            # clue but different facts are two different sentences.
            and _same_evidence(groups[-1].leaning_on, step.leaning_on)
        )
        if merged:
            groups[-1].steps.append(step)
        else:
            groups.append(Group(because=step.because, steps=[step],
                                leaning_on=list(step.leaning_on)))

    return groups


def _same_evidence(one, other):
    """True if two steps leaned on exactly the same earlier steps. Compared by
    identity, since Steps are mutable and two different steps can look alike."""
    return {id(s) for s in one} == {id(s) for s in other}


def to_ai_payload(groups, constraints=()):
    """Turn a grouped trace into plain dicts an AI can be handed to write prose.

    Two rules shape this, and both exist to keep the AI's job to WORDING only:

    - Everything a group needs is written out in full. `because` carries the
      actual fact ("the nation at position 2 is not Japanese"), not just an id.
      If the AI had to chase ids to find out what a step said, it would
      eventually get it wrong.
    - Every group has a stable id. Whatever comes back must use each id exactly
      once, which is how a dropped, invented or silently merged step is caught.
      The wording cannot be checked; the structure can.
    """
    clue_numbers = {}
    for number, clue in enumerate(constraints, 1):
        _index_clue(clue, number, clue_numbers)
    return _payload_for(groups, clue_numbers)


def _index_clue(clue, number, found, top=None):
    """Map a clue AND everything nested inside it to the same clue number and
    the same original sentence.

    An And or Or eliminates through its children, so the step that gets
    recorded names a child. Without this, a step from inside a combined clue
    could not say which of the user's sentences it came from.
    """
    top = clue if top is None else top
    found[id(clue)] = (number, getattr(top, "source_text", None))
    for child in getattr(clue, "constraints", ()):
        _index_clue(child, number, found, top)


def _payload_for(groups, clue_numbers):
    """One level of the trace as dicts. Recurses for sub-proofs."""
    # Which group each step ended up in, so dependencies can point at groups
    # rather than at steps the AI never sees.
    owner = {}
    for number, group in enumerate(groups, 1):
        for step in group.steps:
            owner[id(step)] = number

    payload = []
    for number, group in enumerate(groups, 1):
        payload.append({
            "id": number,
            "kind": group.kind.value,
            "says": [describe(step) for step in group.steps],
            "facts": [
                {"category": step.category, "position": step.position, "value": step.value}
                for step in group.steps
            ],
            "clue": _clue_reference(group.because, clue_numbers),
            "because": _dependencies(group, number, owner),
            "children": _sub_proofs(group, clue_numbers),
        })
    return payload


def _dependencies(group, number, owner):
    """The other groups this one leaned on, each carrying its fact in full."""
    found = []
    seen = set()
    for step in group.steps:
        for dependency in step.leaning_on:
            other = owner.get(id(dependency))
            # Skip anything inside this same group, and anything from a
            # different level of the trace.
            if other is None or other == number or other in seen:
                continue
            seen.add(other)
            found.append({"id": other, "says": describe(dependency)})
    return found


def _sub_proofs(group, clue_numbers):
    """The pretend worlds behind this group, each grouped in its own right."""
    proofs = []
    for step in group.steps:
        for sub in step.children:
            proofs.append({
                # Which of this group's facts this pretend world was run for.
                # A group can merge several removals, each with its own set of
                # sub-proofs, and without this they arrive as one flat pile
                # with nothing to say what each one was arguing about.
                "for": describe(step),
                "assuming": _about_reference(sub.about, clue_numbers),
                "ended_in_contradiction": sub.refuted,
                "steps": _payload_for(group_steps(sub.steps), clue_numbers),
            })
    return proofs


def _clue_reference(because, clue_numbers):
    """Name the reason in terms a person recognises: the user's own sentence
    where there is one, otherwise the puzzle rule that fired."""
    if isinstance(because, Rule):
        return {"kind": "rule", "text": because.value}
    if because is None:
        return None

    number, text = clue_numbers.get(id(because), (None, None))
    return {
        "kind": "clue",
        "number": number,
        # The user's original English, kept on the constraint at translation
        # time. Falls back to the structure only if nothing was recorded.
        "text": text or getattr(because, "source_text", None) or repr(because),
    }


def _about_reference(about, clue_numbers):
    """What a sub-proof was assuming: a shaving assumption is a Step, an Or
    branch is a whole clue."""
    if isinstance(about, Step):
        return describe(about)

    reference = _clue_reference(about, clue_numbers)

    # An Or's branches all share the parent's number and the parent's sentence,
    # so the citation alone cannot tell two branches apart - every branch of
    # "the Norwegian lives next to the blue house" cites that same sentence.
    # Add what THIS branch actually supposes, which is the only thing that
    # distinguishes them.
    if reference is not None and hasattr(about, "describe"):
        reference = dict(reference, says=about.describe())
    return reference
