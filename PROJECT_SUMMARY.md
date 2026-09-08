# Constraint Solver with Proofs — Complete Technical Summary

**Repo:** github.com/sid-chowti/constraint-solver-with-proofs
**Deployed:** Render (free tier web service, Infrastructure-as-Code via `render.yaml`) — *(paste your live `*.onrender.com` URL here; it isn't recorded in the repo)*
**Timeline:** 2026-08-07 → 2026-09-06, 56 commits, solo author
**Stack:** Python 3.11, FastAPI, Uvicorn (ASGI), Pydantic, Anthropic SDK, pytest, vanilla HTML/CSS/JS (no framework, no build step)
**Size:** ~2,880 lines of production Python + ~3,570 lines of tests + 727 lines of front-end. **301 tests, all passing.**

---

## 1. One-paragraph elevator

A full-stack web application that accepts an Einstein/Zebra-style logic puzzle written in ordinary English and returns a guaranteed-correct solution together with a machine-checked, step-by-step deduction proof. A large language model is used **only** as a natural-language front end: it translates prose into a structured constraint representation and never performs a single step of reasoning. All inference is done by a hand-written constraint-satisfaction solver implementing arc consistency and Singleton Arc Consistency, with no search, no backtracking, and no guessing anywhere in the codebase. The architectural thesis is a **trust boundary**: a language model is unreliable at multi-step constraint logic, and a confidently wrong answer is worse than no answer — so language handling and reasoning are separated by a layer of independent validators, and the one class of error no validator can catch is escalated to a human-in-the-loop confirmation screen.

---

## 2. System architecture and data flow

```
English text
   |
   v  (LLM — the ONLY model call in the running app)
Structured JSON
   |
   |--> Guard 1  parsing.clues_from_json           — is this well-formed clue data?
   |--> Guard 2  constraints.validate_constraints  — does the puzzle define what it names?
   |--> Guard 3  grounding.find_ungrounded         — did the user actually write this value?
   |       (all three: failure -> complaint fed back to the model -> retry, max 3 attempts)
   |
   v
HUMAN CONFIRMATION SCREEN  -- the one check no code can perform
   |
   v  (clues carried back up by the browser; server is stateless)
SOLVER  (pure algorithm, zero AI)
   |  arc consistency -> fixed point -> Singleton Arc Consistency (shaving)
   v
Answer + deduction trace (nesting tree + dependency DAG)
   |
   |--> Guard 4  verify.verify        — does the answer satisfy every clue?    (raises: solver bug)
   |--> Guard 5  verify.verify_trace  — does the proof replay to same state?   (raises: solver bug)
   |
   v
Interactive step-through proof viewer in the browser
```

Guards 1–3 are **recoverable** (translation error → retry the model). Guards 4–5 can only indicate a defect in the author's own code, so they raise loudly rather than degrade.

---

## 3. Layer 1 — The constraint solver core (the technical centerpiece)

### 3.1 Problem formalization

The puzzle is modeled as a **Constraint Satisfaction Problem (CSP)**:

- **Variables:** each `(category, position)` cell — e.g. `(color, house 3)`.
- **Domains:** the legal values of that category.
- **Implicit global constraint:** each category is a **bijection** between values and positions — equivalent to an `alldifferent` constraint in both directions.
- **Explicit constraints (from the puzzle's clues):**
  - `AbsolutePosition` — a **unary** constraint: `position(a) ⊙ k` for `⊙ ∈ {==, !=, <, >, <=, >=}`
  - `RelativePosition` — a **binary** constraint: `position(a) + offset ⊙ position(b)`
  - `And` — n-ary conjunction
  - `Or` — n-ary disjunction

Constraint classes are an ABC hierarchy (`Constraint` with abstract `propagate` and `describe`), implemented as Python `@dataclass`es. Operator strings are mapped to Python's `operator` module functions through a table, and validated at construction (`__post_init__`) rather than at use time.

### 3.2 The domain store: `PossibilityGrid`

A candidate-set store mapping `(category, position) → set(values)`, plus a **dual index** (`positions_for(category, value)`) giving the mirror view: which positions a value can still occupy. Both views are needed because the two `alldifferent` propagators run in opposite directions.

Two design points worth calling out in an interview:

1. **The trace lives inside the domain store, not beside it.** `eliminate()` is the single mutation point, so *recording is structurally unavoidable* — a deduction physically cannot occur unobserved. An external tracker would depend on every call site remembering to notify it.
2. **`copy()` deliberately does not copy the step list** — a clone starts with an empty `steps` list but *inherits* the `_killed_by` dependency index. That single decision is what makes the proof come out as a tree with no manual indentation bookkeeping: a hypothetical world naturally accumulates its own sub-proof while still being able to cite facts established before the hypothesis began.

`eliminate()` also detects two derived events automatically:
- domain wipe-out → records a `CONTRADICTION` step *then* raises (so the contradiction is the last line of the sub-proof, not lost in the unwind),
- domain reduced to a singleton → records a `CONCLUDE` step, which is what lets the proof say "therefore it must be X" instead of only ever saying "not this, not that."

### 3.3 Propagators

**Arc consistency (`RelativePosition.propagate`).** Textbook AC `revise`, applied in both directions: a position survives for variable *a* only if **at least one** surviving position of *b* forms a legal pair. Pruning `a` first, then `b` against the already-narrowed `a`, squeezes more out of a single pass.

**The two `alldifferent` propagators (`rules.py`).** These are the standard singles rules of exact-cover-style puzzles, derived from the bijection rather than hard-coded per puzzle:
- `apply_value_used_once` — **naked single**: a cell down to one value removes that value from every other position in its category.
- `apply_value_must_be_somewhere` — **hidden single**: a value with only one remaining position claims it, and also detects the mirror failure (a value with *no* remaining position) that domain wipe-out cannot see.

**Constructive disjunction (`Or.propagate`).** Each disjunct is propagated to a fixed point on its **own copy** of the domain store. Branches that derive a contradiction are refuted; the surviving branches' domains are **unioned**, and any value no surviving branch still permits is eliminated from the parent store. This is exactly constructive disjunction, and it is how "the Norwegian lives next to the blue house" (an `Or` of two `RelativePosition`s) does real work without guessing. Note the asymmetry with shaving below: here a *surviving* branch is load-bearing evidence ("the value died under A **and** under B"), so its sub-proof is retained rather than discarded.

**Fixed-point iteration (`propagate_until_stable`).** Round-robin over all constraints plus both puzzle rules, repeating until a complete pass eliminates nothing. Termination is guaranteed because domains only ever shrink over a finite lattice. Constraints are partitioned by `is_speculative()` into a cheap phase and an expensive phase: all non-`Or` constraints run to a fixed point first, then everything runs together. **Ordering cannot affect the final fixed point** (the loop only exits when nothing at all remains to remove) — it affects *cost*, because `Or` copies the entire grid once per branch, so pre-narrowing makes those branches smaller and kill them sooner. It also makes the resulting proof read the way a person solves: easy clues first, case analysis last.

### 3.4 Shaving = Singleton Arc Consistency (SAC)

Plain arc consistency **stalls on the real Zebra puzzle with 18 of 25 cells still open** (this is asserted by a dedicated test, so the justification for the next algorithm is encoded in the test suite).

`shave()` implements **proof by refutation**: assume a single candidate is true, pin it, propagate to a fixed point on a throwaway copy, and if the puzzle explodes, that candidate was provably never possible — so remove it from the real grid.

Three properties that matter:

- **Refutation-only.** A trial that *survives* proves nothing (other candidates might survive too), so the entire trial is discarded. This is precisely what keeps every conclusion a **proof rather than a guess**, and it is why the explanation is checkable line by line.
- **One level deep, non-recursive.** `deduce_until_stable` alternates full propagation and shaving until neither finds anything; shaving calls `propagate_until_stable` (never itself), which both prevents infinite recursion and keeps every refutation a short chain a human can follow.
- **This is Singleton Arc Consistency**, a named consistency level strictly stronger than arc consistency. The practical payoff, verified experimentally: **human puzzle techniques fall out of it for free instead of being hand-coded.** A "naked pair" (houses 1 and 2 both reduced to `{red, green}`) is invisible to AC and to both puzzle rules, but shaving eliminates red and green from houses 3 and 4 anyway — assuming `red@3` forces both houses 1 and 2 to green, which the `alldifferent` propagator explodes on. One general principle replaces a pile of special cases.

**There is no backtracking and no guessing anywhere in the repository.**

### 3.5 Honest trichotomy of outcomes

`solve()` always returns a `Solution` (never throws for a puzzle-level condition), carrying `Status ∈ {SOLVED, UNSOLVABLE, INCOMPLETE}`:

- `SOLVED` — every cell forced; a full assignment is extracted.
- `UNSOLVABLE` — the clues contradict each other; the partially-solved grid is deliberately returned so the user can see how far it got before the collision.
- `INCOMPLETE` — deduction exhausted with choices open. **The system honestly reports incompleteness rather than guessing**, and states plainly that this means either multiple solutions exist or the puzzle needs search.

Critically, `InvalidConstraint` (a clue naming something the puzzle never defined) still **raises** and is *not* mapped to `UNSOLVABLE` — they are semantically different failures requiring different responses: one is shown to the user, the other triggers an LLM retry. Conflating them would tell users their valid puzzle was impossible.

A trace is produced for **all three** outcomes — a user whose puzzle is contradictory needs the explanation most of all.

---

## 4. Layer 2 — The deduction trace (proof generation)

### 4.1 Two distinct data structures in one record

Each `Step` carries `kind ∈ {ELIMINATE, CONCLUDE, SUPPOSE, CONTRADICTION}`, the cell it concerns, its justification (`because` — either the user's clue object or a `Rule` enum member), and two structurally different link sets:

- **`children` — the nesting TREE.** The sub-proof from inside a hypothetical world (a shaving refutation or an `Or` branch). Its purpose is epistemic: facts that were true only inside an assumption must never be presentable as established facts.
- **`leaning_on` — the dependency DAG.** *Not* a tree: one step can feed many later steps and rest on many earlier ones. This is what lets a step say "because of step 12."

Dependencies are **looked up, never hand-tracked** — the grid already knows which step killed which candidate (`killed_by`), so each propagator queries the index rather than maintaining its own bookkeeping. This removes an entire category of "the proof says the wrong reason" bugs by construction.

### 4.2 Proof minimization — `relevant()`

A **backward reachability walk (DFS with a visited set) over the reversed dependency edges** starting from the target step. Everything unreachable was true but irrelevant. Applied to every sub-proof so a refutation shows only the chain that actually reached the contradiction, not the incidental work the trial did along the way. This is the same idea as extracting an unsatisfiable core.

### 4.3 Proof compression — `group_steps()`

Adjacent `ELIMINATE` steps sharing **both** the same justification **and** the same support set are merged into one `Group`. "Remove red from houses 2, 3, 4, 5" is four steps but one thought. Support-set identity is compared by object identity (`id`), because two distinct `Step`s can be structurally equal. `CONCLUDE`, `SUPPOSE` and `CONTRADICTION` never merge — each is a distinct thought.

**Effect on the Zebra puzzle: 125 raw steps → 74 grouped steps.**

### 4.4 Serialization — `to_ai_payload()`

Converts the grouped trace into plain dicts with two invariants:
- **Every group is self-contained**: dependencies carry the *full fact text*, not just an id, so no consumer ever has to chase references to know what a step said.
- **Every group has a stable integer id**, per level. This id set is the enforcement mechanism for the narration contract in §8.

Clue indexing recurses into `And`/`Or` children so that a step recorded against a nested sub-clue can still cite the top-level sentence the user actually wrote.

---

## 5. Layer 3 — Independent verification

Two verifiers act as **independent oracles**; neither shares code with the solving loop's orchestration.

**`verify(puzzle, constraints, assignment)`** re-reads the finished answer and checks it directly:
- **shape check** — every category is a genuine permutation of its values (catches half-finished or duplicated answers before any clue is consulted),
- **satisfaction check** — every clue holds, evaluated through each constraint's *own* semantics (`RelativePosition.allows`, the operator table).

Because it reuses the clue's own definition of meaning, it catches **orchestration** bugs — a clue that never ran, a botched grid copy, a misread answer — rather than a misunderstanding baked into the clue itself. A failure here raises `VerificationFailed`, because it is always a solver defect and never a fact about the puzzle.

**`verify_trace(puzzle, steps, finished)`** — a **replay equivalence check**. It re-executes only the top-level `ELIMINATE` steps on a fresh empty grid and asserts the result is identical, cell by cell, to what the solver actually ended with. This catches three defect classes invisible to `verify()` because they leave the final answer untouched: a step recorded against the wrong cell, a step lost, a step recorded twice. The reasoning behind it: *a trace is a claim about what the solver did, and an unchecked claim is worth little.*

Nested steps are deliberately not replayed — they happened in discarded hypothetical worlds and never touched the real grid.

The browser rebuilds its display grid by replaying eliminations from an empty start too, which is **the same operation `verify_trace` performs server-side** — so the picture on screen provably cannot drift from what the solver did.

---

## 6. Layer 4 — The AI translation layer and its guards

### 6.1 The contract

`translate.py` calls `claude-opus-5` with a strict system prompt: *never solve, never omit a clue for being redundant, never invent a clue, output one JSON object.* Adaptive extended thinking is enabled (`thinking={"type": "adaptive"}`, `max_tokens=16000`) because translation is the single step whose mistakes nothing downstream can catch.

**Dependency-injection for testability — the key engineering decision here.** `translate(text, ask)` takes an `ask` *function* rather than calling the API itself. Every retry path, every guard rejection, and every malformed-reply branch is therefore exercised with canned replies and **no API key and no cost**. This is what makes the most interesting half of the file testable at all.

### 6.2 The retry loop

Guards are the judge; the model only proposes. On rejection, **every complaint found is collected and fed back at once** (not one per round-trip), numbered, as an instruction to fix all of them. `MAX_ATTEMPTS = 3`, chosen deliberately: since guards report exhaustively, attempt 2 already has the full list and attempt 3 covers one bad round — more just burns money.

### 6.3 Guard 1 — structural parsing (`parsing.py`)

Untrusted JSON → real `Constraint` objects. Trusts nothing, guesses nothing:
- unknown clue type, unknown operator, missing or wrongly-typed field, `[category, value]` pairs that aren't two strings;
- **rejects unknown fields** rather than ignoring them, on the grounds that silently dropping something the model was trying to express is worse than complaining;
- **`isinstance(x, bool)` guard on integer fields** — in Python `True` *is* an `int`, so a naive check would accept `{"position": true}` and silently read it as house 1;
- **recursion depth cap (`_MAX_DEPTH = 10`)** — without it a deeply nested payload blows the Python stack (a crash, and a denial-of-service vector) instead of producing a readable complaint;
- **collects all problems before raising**, so one retry fixes everything.

Explicitly *not* done here: existence checking (needs the puzzle — that's Guard 2) and fuzzy matching (`"Purple" → "purple"`). Normalization belongs *above* the trust boundary where being wrong is recoverable; below it, a guess becomes a confident wrong answer.

`clue_to_json` is the inverse direction, used for persistence and for a **round-trip property test** that checks the parser against itself.

### 6.4 Guard 2 — referential validation (`constraints.validate_constraints`)

Walks every clue (recursing into `And`/`Or`) and checks every `(category, value)` pair and every literal position against what the puzzle actually defines, emitting structured `BadReference` records (`kind`, `category`, `value`, `allowed`) — **pieces, not prose**, precisely so the retry loop can machine-read `allowed` and feed it back.

Catching an out-of-range position here rather than downstream matters: propagation would cross the value off everywhere and report the *puzzle* as unsolvable, when in fact the *clue* is broken — two conditions requiring opposite responses.

### 6.5 Guard 3 — lexical grounding (`grounding.py`) — the only guard that checks the model against the *user*

Guards 1 and 2 both check the model **against itself**, and a hallucinated value is perfectly consistent with the rest of a hallucinated puzzle — neither can see it. Grounding asks a different question: *does every value the model listed actually appear in the text the user typed?*

Deliberately **one-sided and permissive**: plain lexical search, no model, no semantics. It can only detect a value pulled from thin air, never a value misunderstood. The permissiveness is a reasoned trade-off — a false alarm blocks somebody whose puzzle was fine and prevents them ever reaching the confirmation screen, whereas a miss still gets seen by a human on that screen.

Making the real Einstein puzzle pass required all three of these:

| value | the text says | why a naive check fails |
|---|---|---|
| `English` | "the **English**man lives…" | value is a prefix of a longer word |
| `OldGold` | "the **Old Gold** smoker…" | the model dropped the space |
| `Kools` | "**Kool** is smoked…" | plural/singular mismatch |

Implementation: CamelCase/space/hyphen/underscore **word segmentation** via regex, re-joined into a pattern tolerating any of those separators, with a **leading word boundary but deliberately no trailing one** (so `English` matches `Englishman`), plus singular candidates (`-s`, `-es` stripping), case-insensitive. Category *names* are never checked — people write "the red house," not "colour."

`hint_for()` translates the model-facing complaints into **human-facing advice**: "never invent a value" is sensible to a model and baffling to a person, and the real-world cause is almost always the same — the user pasted the clues but omitted the line listing the options — so the app says exactly that.

### 6.6 A deliberate non-guard: warnings

The same value appearing in two categories is *usually* the model confusing itself, but can be legitimate ("Green" as both a colour and a surname). It is therefore surfaced as a **warning on the confirmation screen and never triggers a retry** — retrying a legitimate puzzle would loop until it gave up.

---

## 7. Layer 5 — Human-in-the-loop verification (the design's sharpest idea)

**The problem no code can solve.** The three guards catch the model contradicting itself. None can catch it *consistently misreading* the user. If it reverses "the green house is immediately to the right of the ivory house," every guard passes, and the solver returns a *guaranteed-correct answer to a puzzle nobody asked*. Detecting that requires understanding English — which is exactly the job the model was hired for, so it cannot also be the check.

So the flow is **split into two endpoints** with a confirmation screen between them, and three rules make that screen worth reading:

1. **What is displayed is generated from the clue's own structured fields, never from the user's sentence.** `Constraint.describe()` deliberately never falls back to `source_text`. Echoing a user's own words back to them is a **mirror, not a check** — it confirms nothing.
2. **Directional clues are stated both ways**, so the reader never has to mentally flip "a is left of b" into "b is right of a" — that flip *is* the mistake being hunted, so the screen must not ask the reader to perform it.
3. **The second reading is produced by mirroring the clue's structure, not its words**: `mirrored()` swaps the operands, negates the offset, and inverts the relational operator, then describes the *result*. Rewriting the sentence (string-swapping "left" and "right") would eventually turn "somewhere right of" into "immediately left of" and invent a constraint the user never wrote. **The mirror is validated against the solver's own `allows()` across every operator × offset × position-pair combination in the test suite.**

The screen also shows the *shape* the model inferred (position count and categories), since a fabricated category is a misreading like any other.

**Any clue can be toggled off before solving.** This is provably safe: removing a constraint can only ever *widen* the solution set, so the outcome is the same answer or an honest `INCOMPLETE` — never a wrong one.

The confirmation deliberately comes **before any answer exists**, so there is nothing for the reader to anchor on and rubber-stamp.

---

## 8. Layer 6 — Trace narration (`narrate.py` / `narrate_example.py`)

Turning the solver's mechanical wording into readable English is the one place where trusting a model looks like it contradicts the whole project. Three properties make it safe:

1. **A structural bijection contract.** Each step group has an id; the reply is a JSON object of `id → sentence` and **must use each id exactly once**. A dropped step, an invented step, or two steps silently merged all change the returned key set, and all are rejected. *The wording cannot be checked; the structure can.*
2. **All-or-nothing.** If the reply doesn't line up, none of it is used — a half-narrated proof is worse than a bare one, because nothing marks the gaps.
3. **The machine-checked fact stays on screen underneath the sentence.** Prose sits *on top of* the proof, never in place of it.

Sub-proofs are intentionally not narrated: ids are numbered per level so a flat reply couldn't address them without collisions, and skipping them cut the payload **from 142 KB to 27 KB**, which is what made this a single API call.

**The subtle failure mode found and guarded — the case-split honesty trap.** An `Or`'s eliminations are recorded against the *whole* clue even when the real work happened inside its branches. So one step of the Einstein proof rules out **water** while citing a clue about nationalities and colours. A writer that couldn't see this would confidently explain that the clue says something it does not. Fix: `for_writing()` sends a `cases` count per step, and the prompt forbids reading such steps off their clue — they must hedge ("whichever way this clue falls…", "assuming otherwise broke the puzzle"). Two Einstein sentences were found guilty of this and rewritten.

**Architectural evolution:** live narration was later removed from the running app entirely. All three bundled puzzles ship **201 hand-checked sentences** committed to the repo, so every demo reads as English with **no API key and no cost**, and `narrate.py`/`narrate_example.py` (an argparse CLI: `--api-key`, `--name`, `--all`) became **maintenance tooling** you point at a puzzle when the stored wording needs regenerating. It refuses to write a file that fails the same structural check the live path applied. **The result: exactly one endpoint in the whole application can reach a model, and it reads English — it does not write it. A test asserts this.**

Three tests prevent hand-written prose from rotting silently:
- the same structural check, re-run against the **live** trace, so wording cannot drift out of step with the solver;
- every sentence is a finished sentence (no stubs, no truncation);
- **every case-split step visibly hedges** — the one thing the structural check cannot catch. The test knows which steps had `cases > 0` and requires a hedge marker in their wording. It was verified to actually bite by temporarily rewording a sentence as a flat reading of its clue.

---

## 9. Layer 7 — Web API (FastAPI)

Deliberately thin: this layer moves data and decides nothing about a puzzle.

| endpoint | purpose |
|---|---|
| `GET /` | serves the single-page app |
| `GET /api/examples` | lists bundled puzzles — **also the allow-list** for the next endpoint |
| `GET /api/example?name=` | one bundled puzzle, already translated and solved. No key, no cost. Skips confirmation on purpose: a human already checked it |
| `POST /api/translate` | English + the **visitor's own** API key → puzzle shape + clues, each stated back in plain words. **Solves nothing** |
| `POST /api/solve` | confirmed clues → answer + full trace. **No AI, no key, no cost** |

**Stateless by design.** Nothing is remembered between the two calls; the browser carries the clues back up. The reasoning is explicit: a server-side pending-puzzle store would need expiry, would leak every abandoned puzzle, would lose them all on restart — and **a free-tier host sleeps whenever nobody is visiting, which is exactly while somebody is reading the confirmation screen.**

That is safe for the same reason the model was never trusted: `/api/solve` runs every guard on whatever it is handed, regardless of origin. And a visitor rewriting their own puzzle **is the feature, not an attack** — there is no privilege to escalate, only their own puzzle to change.

**Error-code discipline:** `400` for data the caller can fix, `404` for an unknown example name, `422` with the full per-attempt complaint list (plus a plain-English `hint`) when translation fails, `502` when the upstream API itself fails. Nothing that is a user-data problem is ever allowed to surface as a `500`.

**Deliberately loose Pydantic model.** `SolveRequest` declares `dict` and `list`, not a precise shape, because `Puzzle._validate`, `clues_from_json` and `validate_constraints` already do that job and produce complaints written for a human. Describing the shape twice would mean two boundaries to keep in sync — and the weaker one would win by running first.

### Security properties

- **Path traversal:** `?name=` is user-controlled, so it is checked against the bundled allow-list and never handed to the filesystem — otherwise `../../secrets` would be a perfectly good example name. Explicitly tested.
- **Bring-your-own-key (BYOK):** the app holds no API credential at all. A public URL on the owner's key is an open tab anyone can run up. The visitor's key is used for a single request, then dropped — never stored, never logged.
- **No database, no secrets, no session state**, so the whole attack surface is the two JSON endpoints.
- **XSS:** every interpolated string in the front end goes through an `esc()` HTML-escaper.
- **DoS:** recursion depth cap on nested clue structures.

---

## 10. Layer 8 — Front end (727 lines, vanilla HTML/CSS/JS, zero dependencies)

Framework choice was made deliberately after research rather than by default: no build step, no toolchain, no `node_modules`, which for a single page is a real advantage in deployability and in being able to explain every line.

**The interactive proof walker** is the substantial piece:

- The trace is a **tree**, not a list, so the viewer maintains a `path` of `{step, branch}` hops plus an index, and reconstructs state on demand.
- **`stateAt(path, index)` replays eliminations from an empty grid** to reconstruct the board as it stood at any moment — the same operation `verify_trace` performs on the server, so the display provably cannot drift from the solver. Descending into a sub-proof rewinds to the world *just before* its parent step ran, which is exactly the grid the sub-proof was copied from.
- The board keeps values in fixed positions and **strikes them through as they die**, so the eye can follow a single cell across steps: values killed by *this* step are picked out in red, and a cell that has just run out of alternatives turns green.
- **You step *into* a hypothetical world rather than seeing it inline.** A single Zebra step can hang 88 pretend worlds beneath it; printing them inline would present facts that were only ever true under an assumption as though they were real. So the grid switches to that pretend world, states plainly that nothing on it is known to be true, says whether the branch ends in a contradiction (which is what refutes the assumption) or survives and agrees with the others, runs to its conclusion, and you back out. Breadcrumbs show the assumption chain.
- **Keyboard navigation:** arrow keys walk the proof, Escape leaves a pretend world, both suppressed while typing in the textarea.
- Accessibility detail: **no `behavior: 'smooth'` anywhere** — smooth scrolling is a no-op when the reader has "reduce motion" enabled, which would silently break every scroll that matters, including bringing an error message into view.
- An `Or` branch label uses the branch's *own* `says` rather than the shared clue text, since every branch of "the Norwegian lives next to the blue house" cites that same sentence and only `says` distinguishes them.

---

## 11. Deployment and DevOps

- **Render**, free-tier web service, defined as **Infrastructure-as-Code** in `render.yaml` (declarative blueprint: runtime, plan, build command, start command, env vars) rather than clicked together in a dashboard.
- **ASGI production server:** `uvicorn app:app --host 0.0.0.0 --port $PORT` — binding all interfaces and reading the platform-injected `$PORT` are both required for containerized hosting.
- **`PYTHON_VERSION` pinned to `3.11.13`** — the *exact* version the project is developed and tested against. The code requires ≥3.10 for `str | None` union syntax, but "newer than 3.10" is a different claim from "tested," and deploying onto a version no test has ever run against is how you find out the hard way.
- **Dependency pinning with documented reasons:** `anthropic>=1.4.0` (major SDK version bump; the code uses `thinking={"type": "adaptive"}`, which older releases reject outright), `fastapi>=0.115.0`, `uvicorn[standard]>=0.32.0`. Dev dependencies (`pytest`, `httpx` for the FastAPI `TestClient`) are split into a separate `requirements-dev.txt` so the deployed image does not carry a test framework.
- **No build step, no database, no server-side secrets** — a consequence of the stateless BYOK design, which makes the deployment genuinely simple.
- Known free-tier characteristics were reasoned about explicitly before shipping: the instance sleeps after ~15 minutes so most visits are cold starts, and a 3-attempt translation could approach a request timeout. The mitigation is that **the three bundled, pre-translated, pre-narrated puzzles carry the entire demo with no key and no upstream call** — which is what makes the site usable for the ~100% of visitors who don't own an Anthropic key.

---

## 12. Testing (301 tests)

| file | tests | covers |
|---|---:|---|
| `test_constraints.py` | 54 | clue semantics, `allows`, mirroring, describe/also_means, validation |
| `test_app.py` | 35 | all five endpoints, error codes, path traversal, the full confirm-then-solve flow |
| `test_solve.py` | 35 | propagation, both rules, phase ordering, order-independence, shaving |
| `test_parsing.py` | 20 | malformed data, every complaint path, JSON round-trip |
| `test_translate.py` | 20 | every retry path, with canned replies and no API key |
| `test_trace.py` | 18 | dependency recording, sub-proofs, backward slicing |
| `test_deduction.py` | 16 | grouping, `relevant()`, payload construction |
| `test_possibilities.py` | 15 | domain store, copy semantics, killed-by index |
| `test_examples.py` | 14 | bundled puzzles: solve, minimality, grounding, prose/trace agreement, hedging |
| `test_narrate.py` + `test_narrate_example.py` | 17 | structural contract, all-or-nothing rejection, CLI |
| `test_verify.py` | 9 | both verifiers |
| `test_grounding.py` | 8 | prefix, spacing, plural cases |
| `test_puzzle.py` | 7 | puzzle validation |
| `test_zebra.py` | 4 | the real Zebra puzzle end to end |

Testing techniques worth naming:

- **Differential testing against an independent oracle.** 1,800 randomly generated puzzles were classified against ground truth computed by **exhaustive permutation enumeration calling only `verify.holds`** — so the oracle structurally *cannot* inherit a solver bug.
- **Property-based invariants** — e.g. clue order must not change the answer; a shaved candidate must never be one a real solution uses; inclusive operators must match the shifted strict ones; JSON round-trips must be identity.
- **Executable design rationale.** `test_plain_propagation_alone_cannot_finish_the_zebra_puzzle` asserts that AC stalls — i.e. it encodes *why shaving exists*. If a future rule ever makes plain propagation strong enough, that test fails, and the comment explains that this is a good failure. A companion test pins the exact deduction AC cannot make (assuming Japanese in house 2 forces tea out, juice in, Lucky Strike into house 2, leaving Parliaments nowhere to go).
- **Anti-drift tests** on committed artifacts: stored prose is re-checked against the *live* trace on every run, so hand-written content cannot rot silently.
- **Negative-control validation:** the case-split hedging test was verified to actually bite by temporarily rewording a sentence to break it.

---

## 13. Measured results

| metric | value |
|---|---|
| Classic Zebra/Einstein puzzle | **solved in ~36 ms** |
| Zebra proof size | 125 raw steps → **74 grouped steps** |
| AC alone on Zebra | **stalls with 18 of 25 cells open** — motivates SAC |
| Randomized soundness suite | **1,800 puzzles** (4×2, 4×3, 5×3) vs. exhaustive ground truth — **perfect classification, zero defects** |
| ↳ 1,024 impossible puzzles | 1,024 correctly `UNSOLVABLE` (100% caught, not merely "not wrong") |
| ↳ 766 ambiguous puzzles | 766 correctly `INCOMPLETE` — **never once falsely claimed `SOLVED`** |
| ↳ candidate elimination errors | **0** — never removed a value used by a real solution |
| Strength, exhaustive ground truth (4×2, 4×3) | AC alone: 99.5% / 99.8% → **with SAC: 100% / 100%, zero stalls** |
| Ceiling probe: **340 minimal** uniquely-solvable puzzles up to 7×5 | **all 340 solved, zero stalls.** Median 2–5 ms, worst 55 ms |
| Total generated puzzles benchmarked | **~2,940, zero defects** |
| Test suite | **301 tests, all passing (~10 s)** |
| Bundled demos | 3 puzzles, all **minimal**, 201 committed prose sentences |

**Minimal puzzles are the hardest of their size** — every clue is load-bearing, with no redundancy to lean on — which is why the ceiling probe used them: built by adding true clues until the solution was unique, then trimming every clue not required for uniqueness.

**A measured conclusion that changed the plan:** the benchmarks established that *solver strength is not the bottleneck*, so adding backtracking or deeper shaving was explicitly rejected as a non-improvement, and effort went to the proof trace instead. What backtracking *would* add is distinguishing the two causes of `INCOMPLETE` ("puzzle is genuinely ambiguous" vs. "solver too weak") — a diagnostic, deliberately deferred.

---

## 14. Bundled demo puzzles

| puzzle | size | clues | grouped steps | prose |
|---|---|---:|---:|---:|
| The Einstein Puzzle | 5 houses × 5 categories | 14 | 74 | 74 sentences |
| The Four Desks | 4 desks × 4 categories | 11 | 53 | 53 sentences |
| The Five Food Trucks | 5 trucks × 4 categories | 13 | 74 | 74 sentences |

All three are **minimal**, enforced by a test that removes each clue in turn and asserts uniqueness is lost. They were **authored against the solver rather than by hand** — the first desks draft had four solutions, and the first trucks draft was ambiguous and then over-specified. Each carries a `position_noun` ("house"/"desk"/"truck") so the UI never calls a food truck a house.

---

## 15. Engineering decisions and trade-offs worth citing

1. **Separation of concerns as a correctness argument, not a style preference.** AI handles language; a deterministic algorithm handles logic. The boundary is enforced by five independent validators, a two-step user flow, and a test asserting only one endpoint can reach a model.
2. **Recoverable vs. unrecoverable failures are never conflated.** Guards 1–3 produce structured, machine-readable complaints and trigger a retry. Guards 4–5 raise, because they can only mean the author's own code is broken. `InvalidConstraint` ≠ `UNSOLVABLE`.
3. **Structured errors, not prose errors.** `BadReference` and `MalformedClue` carry `allowed` lists as data specifically so the retry loop can feed them back — digging that out of a formatted sentence would be miserable. All problems are collected before raising, so one retry fixes everything.
4. **Recording made structurally unavoidable.** Because the domain store owns the trace and `eliminate()` is the only mutation point, an unrecorded deduction is impossible by construction rather than by discipline.
5. **Dependencies looked up, never hand-tracked.** The grid already knows who killed what, so propagators query it rather than maintaining parallel bookkeeping.
6. **Testability designed in.** Injecting `ask` as a function is what makes the entire LLM retry surface testable without a key or a dollar.
7. **A check must not be a mirror.** `describe()` refuses to fall back to the user's own words; the second reading is derived by structural transformation, not string editing. This is the single most subtle idea in the project.
8. **Refutation-only inference.** Discarding surviving trials is a deliberate loss of power in exchange for the guarantee that every conclusion is a proof — which is the only reason a line-by-line checkable explanation exists at all.
9. **Reasoned incompleteness.** The system says "I could not finish and here is honestly why" rather than guessing. Toggling a clue off is provably monotone (it can only widen the solution set), so the UI cannot produce a wrong answer.
10. **Statelessness chosen from platform constraints**, not dogma: free-tier sleep would destroy any in-memory pending store precisely while a user is reading the confirmation screen.
11. **Prompt hardening against a discovered failure mode.** The `Or`-attribution trap (a step killing *water* while citing a clue about nationalities) was found empirically, guarded by passing a `cases` count, and locked down by a test that requires visible hedging.
12. **Committed artifacts get anti-drift tests.** Anything hand-written and checked in is re-validated against live behavior on every test run.

---

## 16. Formal terminology index (for the résumé)

Constraint Satisfaction Problem (CSP) · constraint propagation · arc consistency (AC) · **Singleton Arc Consistency (SAC)** · domain filtering / domain store · `alldifferent` / bijection constraints · naked singles and hidden singles as derived consequences · **constructive disjunction** · fixed-point iteration over a monotone finite lattice (termination proof) · solution-preserving pruning (soundness) · proof by refutation / proof by contradiction · no backtracking, no search, no heuristics · dependency DAG · backward reachability / DFS proof slicing (unsat-core-style minimization) · trace grouping by justification-and-support equivalence · replay-equivalence verification · independent oracle testing · differential testing against exhaustive permutation enumeration · property-based invariants · trust boundary / input validation at a boundary · structured error objects for automated repair · LLM output validation and constrained retry loops · structural (bijective-id) contracts on LLM output · hallucination detection via lexical grounding · human-in-the-loop verification · REST API design · stateless server architecture · ASGI · Pydantic validation · Infrastructure-as-Code · BYOK credential model · path-traversal allow-listing · XSS escaping · recursion-depth DoS mitigation.

---

## 17. Skills demonstrated (résumé-relevant inventory)

- **Algorithms:** implemented a constraint solver from scratch — AC propagation, dual-view domain stores, constructive disjunction, SAC/shaving, fixed-point iteration with proven termination and soundness; graph algorithms (DFS reachability over a dependency DAG) for proof minimization.
- **Systems design:** layered architecture with an explicit trust boundary; five independent validators with distinct failure semantics; stateless API shaped by real platform constraints.
- **Testing & verification:** 301 tests; differential testing against an independent exhaustive oracle; ~2,940-puzzle benchmark with zero defects; property-based invariants; tests that encode design rationale; negative-control validation of a test's sensitivity.
- **Applied AI engineering:** structured-output extraction, guard-driven retry loops with machine-readable complaints, hallucination detection, structural contracts on generative output, dependency injection for zero-cost testing of an LLM integration, prompt hardening against an empirically discovered failure mode, and a deliberate reduction of the model's role from the running critical path to offline tooling.
- **Full-stack web:** FastAPI + Pydantic REST API, dependency-free single-page front end with a non-trivial interactive tree-walking visualization, keyboard accessibility and reduced-motion handling, XSS-safe rendering.
- **DevOps:** Render deployment via declarative IaC, ASGI production serving, exact-version runtime pinning with documented reasoning, split runtime/dev dependencies.
- **Engineering judgment:** measured before optimizing and *declined* to add power the data showed was unnecessary; documented every non-obvious trade-off in code comments and a substantial README.

---

## 18. Two things to fill in / caveat before using this

1. **The live Render URL** is not recorded anywhere in the repo — add it at the top.
2. **The ~2,940-puzzle benchmark scripts are not committed** (they were throwaway scripts). The results are real and reproducible, but a reader cannot run them from the repo today. Decide how you want to phrase that.
