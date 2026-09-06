# Logic Puzzle Solver

Type an Einstein/Zebra-style logic puzzle in plain English. An AI turns the
sentences into structured clues. A hand-built solver works out the answer and
shows a step-by-step chain explaining why each conclusion follows.

**The AI never solves anything.** It reads language and nothing else. Every
piece of reasoning is done by the solver, which is a plain algorithm with no
model in it anywhere — because a language model cannot be trusted to reliably
chain multi-step constraint logic, and a wrong answer that sounds confident is
worse than no answer.

```
English -> AI -> JSON -> parser -> validator -> YOU -> SOLVER -> trace -> page
           |              |          |           |       |
      language       "is this    "does the   "is this  all the reasoning,
        only          a clue?"    puzzle      really    and a proof of
                                  have        my        every step
                                  these?"     puzzle?"
```

## What it guarantees

Three independent guards, each catching what the others cannot:

| guard | catches |
|---|---|
| `parsing.clues_from_json` | data that is not a clue at all — bad type, bad operator, missing field |
| `constraints.validate_constraints` | a clue naming a category, value, or position this puzzle does not have |
| `verify.verify` | a finished answer that breaks one of its own clues |
| `verify.verify_trace` | an explanation that does not match what the solver actually did |

The first two objections are fed back to the AI for a retry — they mean the
translation was wrong, not that the puzzle is impossible. The last two can only
mean a bug in this code, so they raise.

## The one check no code can do

The three guards above catch the AI contradicting itself. None of them can
catch it *consistently* misreading you. If it turns "the green house is
immediately to the right of the ivory house" around, every guard passes and the
solver returns a guaranteed-correct answer to a puzzle you never asked. Spotting
that needs someone who understands English — which is the exact job the AI was
hired for, so it cannot also be the check.

So the site asks you. Translating and solving are two separate steps, and in
between it shows every clue it read:

```
"The green house is immediately to the right of the ivory house."
  -> ivory (color) is immediately left of green (color)
  -> so green (color) is immediately right of ivory (color)
```

Three rules make that screen worth reading:

- **What it shows is built from the clue's data, never from your sentence.**
  Echoing your own words back would be a mirror, not a check.
- **Direction clues are said both ways**, so you never have to flip "left of"
  into "right of" in your head — that flip is the mistake being hunted.
- **The second reading comes from mirroring the clue's structure** (swap the
  sides, negate the offset, reverse the operator) and describing that. Rewriting
  the words would eventually turn "somewhere right of" into "immediately left
  of" and invent a constraint. The mirror is checked against the solver's own
  `allows()` over every operator, offset and pair of positions.

Any clue can be switched off before solving. That can only ever *widen* the set
of valid answers, so the result is the same answer or an honest `incomplete` —
never a wrong one.

Nothing is remembered between the two steps. The browser carries the clues back
up, which is safe for the same reason the AI was never trusted: `/api/solve`
runs every guard on whatever it is handed, and a visitor editing their own
puzzle is the feature.

## How the solver works

Constraint propagation, not search. Each clue repeatedly crosses off candidates
it can prove impossible, until nothing more falls out.

That alone stalls on the real Zebra puzzle with 18 of 25 cells still open, so it
also does **shaving**: assume a candidate, propagate, and if the puzzle explodes
then that candidate was never possible. Shaving only ever *refutes* — a trial
that survives proves nothing and is discarded. That refusal is what keeps every
conclusion a proof rather than a guess, and it is why the explanation can be
checked line by line.

There is no backtracking and no guessing anywhere in the repository.

### Measured

- Solves the classic Zebra puzzle in ~35 ms
- 2,940 generated puzzles, checked against brute-force ground truth: never a
  wrong answer, never a solvable puzzle called impossible, never a stall on a
  uniquely-solvable one
- 340 *minimal* uniquely-solvable puzzles up to 7x5 — all solved, none stalled

Shaving is essentially Singleton Arc Consistency, which means human techniques
like naked pairs fall out of it for free rather than being coded one by one.

## The explanation

Every removal is recorded with its reason and the earlier steps it leaned on.
Two structures come out of that:

- **nesting** (a tree) — what happened inside an assumption, so facts that were
  only true in a hypothetical cannot be mistaken for real ones
- **dependencies** (a graph) — what each step leaned on, so a chain can say
  "because of step 12" and so a backwards walk can trim a proof to what mattered

Neighbouring steps sharing a reason and its evidence are grouped, so the Zebra
puzzle reads as 74 steps rather than 125.

You walk those 74 one at a time, not as a wall of text. Each step shows the
fact, the clue it came from, and the board as it stood at that moment — values
struck through as they die, the ones this step killed picked out in red, and a
cell that has run out of alternatives in green. Arrow keys move.

The board is rebuilt in the browser by replaying eliminations from an empty
start, which is the same thing `verify_trace` does on the server, so the picture
cannot drift from what the solver actually did.

Sub-proofs are the interesting part. A step can hang 88 pretend worlds under
itself, and printing them inline would show facts that were only ever true
inside an assumption as though they were real. So you step *into* one: the grid
switches to that pretend world, says plainly that nothing on it is known to be
true, runs to its contradiction, and you back out.

## Running it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --reload
```

Then open http://127.0.0.1:8000.

The bundled Einstein puzzle ships already translated, so it solves with **no API
key and no cost**. A key is only needed to translate a new puzzle from English,
and the site asks the visitor for their own — it is used for one request and then
dropped, never stored and never logged.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
```

236 tests. The translator's retry paths and the whole confirm-then-solve flow
are covered with canned replies, so everything runs without an API key.

## Layout

```
puzzle.py       the shape of a puzzle: categories, values, how many positions
deduction.py    recorded steps, and the tools for grouping and trimming them
possibilities.py the grid of what is still possible, and the trace of why
rules.py        the two facts true of every puzzle, whatever the clues say
constraints.py  the clue types, and the guard on what they may name
verify.py       independent checks on the answer and on the explanation
solve.py        propagation, shaving, and the public solve() front door
parsing.py      untrusted data -> real clues, or a clear list of complaints
translate.py    English -> clues, with the guards as the judge
app.py          the web layer
```
