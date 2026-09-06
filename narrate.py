"""Turn the solver's mechanical steps into sentences a person would write.

This is the LAST place the AI appears, and the most tightly fenced. The solver
has already done every piece of reasoning and proved it; nothing here may add,
drop, reorder or reinterpret a single deduction. The AI is a writer, not a
thinker - handed a finished proof and asked only to say it better.

    "the smoke at position 3 is not Kools"
        ->  "House 3 can't be the Kools smoker: we already know it isn't the
             yellow house, and clue 7 puts Kools in the yellow house."

WHY THIS IS SAFE, when trusting an AI with the explanation sounds like exactly
what this project refuses to do: the wording cannot be checked, but the
STRUCTURE can. Every group carries an id, and the reply must use each id
exactly once. A dropped step, an invented step, or two steps quietly merged all
change which ids come back, and all get caught and sent back for another go.
The reader still sees the checked facts underneath; prose sits on top of them.

Sub-proofs are deliberately not narrated. Ids are numbered per level, so a flat
reply could not address them without collisions, and they are detail a reader
opens on purpose - the mechanical wording is fine there. Dropping them also
takes the payload from 142 KB to 27 KB, which is what makes this one call.
"""
import json
import re

MODEL = "claude-opus-5"

# Two is enough. The only rejection is structural - wrong ids - and a model
# that cannot return the right keys twice will not manage it on a third go.
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = """You put a finished logic-puzzle proof into plain English. You do NOT solve anything.

Every deduction has already been made and machine-checked. Your only job is
wording. Never add a deduction, never leave one out, never merge two, and never
change what one says - even if you think the proof could be shorter or a step
is obvious. It is not your proof.

You get a JSON list of steps. Each has:
  "id"      a number you must use exactly once in your reply
  "says"    the facts it established, in mechanical wording
  "clue"    the user's own sentence, or the puzzle rule, that forced it
  "because" earlier steps it leaned on, with their ids and what they said
  "cases"   how many possibilities had to be checked to prove it (often 0)

Reply with ONE JSON object and nothing else: the id as the key, your sentence
as the value.

  {"1": "The green house can't be house 1...", "2": "..."}

Rules for the writing:
- One or two sentences per step. Plain words. No jargon, no "we can deduce
  that", no step numbers in the prose.
- Positions are houses: say "house 3", never "position 3".
- Say WHY, using the clue and the earlier facts given to you. A step with no
  "because" follows from its clue alone - say that.
- When a step lists several facts, cover them together rather than one by one.
- Quote the user's own clue wording when it helps, in double quotes.
- A "conclude" step means a house has run out of other options. Say that: the
  other possibilities are gone, so this is what is left.

CRITICAL - steps where "cases" is greater than 0. That step was NOT read
straight off its clue. It was settled by checking every way things could be and
finding they all point the same way, or by assuming the opposite and watching
the puzzle fall apart. The clue named is where the case split came from, not a
sentence that plainly says the conclusion.

For these, never write "clue 14 says the Norwegian is next to blue, so house 2
isn't water" - the clue says no such thing about water. Write "whichever side of
the Norwegian the blue house is on, house 2 can't be water" or "assuming house 2
were water broke the puzzle". If the clue's own words do not plainly give the
conclusion, do not pretend they do."""


class NarrationFailed(Exception):
    """The prose could not be trusted, so none of it is used. Carries
    `.attempts`, the structural complaint from each try."""

    def __init__(self, attempts):
        self.attempts = attempts
        super().__init__("; ".join(attempts[-1]) if attempts else "no attempts")


# Strip a trace down to what a writer needs: no sub-proofs, no raw facts.
#
# `cases` is the one thing that must survive from the sub-proofs. A clue like
# "the Norwegian lives next to the blue house" is an Or, and its eliminations
# are recorded against the whole clue even when the real work happened inside
# the branches - so step 35 of the Einstein puzzle rules out WATER while citing
# a clue about houses and nationalities. A writer who cannot see that would
# confidently explain that the clue says something it does not. The count says
# "this was proved by cases, do not read it off the clue".
def for_writing(groups):
    return [
        {"id": group["id"], "kind": group["kind"], "says": group["says"],
         "clue": group["clue"], "because": group["because"],
         "cases": len(group.get("children") or [])}
        for group in groups
    ]


# Check a reply addresses exactly the steps it was given, once each.
def problems_with(prose, groups):
    if not isinstance(prose, dict):
        return ["Your reply must be a JSON object of id -> sentence."]

    wanted = {str(group["id"]) for group in groups}
    given = set(prose)
    problems = []

    missing = sorted(wanted - given, key=int)
    if missing:
        problems.append(f"You left out these step ids: {', '.join(missing)}. "
                        "Every step needs a sentence, including ones that look obvious.")

    extra = sorted(given - wanted)
    if extra:
        problems.append(f"These ids are not steps in this proof: {', '.join(extra)}. "
                        "Use only the ids you were given.")

    empty = sorted((key for key in given & wanted if not str(prose[key]).strip()), key=int)
    if empty:
        problems.append(f"These steps got an empty sentence: {', '.join(empty)}.")

    return problems


# English for every top-level step, keyed by id. Raises NarrationFailed.
def narrate(groups, ask, max_attempts=MAX_ATTEMPTS):
    steps = for_writing(groups)
    if not steps:
        return {}

    conversation = [{"role": "user", "content": _first_request(steps)}]
    attempts = []

    for _ in range(max_attempts):
        reply = ask(conversation)
        prose = _extract_json(reply)
        problems = problems_with(prose, steps)

        if not problems:
            return {int(key): str(value).strip() for key, value in prose.items()}

        attempts.append(problems)
        conversation = conversation + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": _retry_request(problems)},
        ]

    raise NarrationFailed(attempts)


def _first_request(steps):
    return ("Put these deduction steps into plain English.\n\n"
            + json.dumps(steps, indent=1))


def _retry_request(problems):
    listed = "\n".join(f"{number}. {problem}"
                       for number, problem in enumerate(problems, 1))
    return ("That reply could not be used. Problems found:\n\n"
            f"{listed}\n\n"
            "Reply again with the corrected JSON object only.")


def _extract_json(reply):
    """Pull the JSON object out of a reply, or None. Same leniency as the
    translator: the wrapper may be sloppy, the contents may not."""
    if not isinstance(reply, str):
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", reply, re.DOTALL)
    candidate = fenced.group(1) if fenced else reply

    if not fenced:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = candidate[start:end + 1]

    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return None


def anthropic_asker(api_key, model=MODEL):
    """An `ask` backed by the real Claude API.

    Imported lazily so everything else works with the anthropic package absent.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def ask(conversation):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            # Wording only - the thinking was all done by the solver. Adaptive
            # still helps it keep 74 dependent steps straight.
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=conversation,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return ask
