"""Regenerate the stored wording for a bundled puzzle.

The prose that ships with each example has to be written once by something that
can write English. Doing that by hand works, but it leaves no way back: change
the solver or the grouping and the stored sentences no longer describe the
steps, and the only repair is writing them all again. This is the way back.

    python -m narrate_example --api-key sk-ant-... --name desks
    python -m narrate_example --api-key sk-ant-... --all

Nothing is written unless the new wording passes the SAME structural check the
live app applies: every step id addressed exactly once. A file that failed that
would be worse than no file, because the demo would confidently narrate steps
that are not there.

This is a maintenance script, not part of the running app - the app only ever
READS what this produces.
"""
import argparse
import json
import pathlib
import sys

from deduction import to_ai_payload
from examples import EXAMPLES, available, load
from narrate import (
    NarrationFailed,
    anthropic_asker,
    for_writing,
    narrate,
    problems_with,
)
from solve import Status, solve

NOTE = ("Plain-English wording for {title}, written once and stored so the demo "
        "needs no API key. Regenerate with: python -m narrate_example "
        "--api-key sk-ant-... --name {name} . Keys are group ids from "
        "to_ai_payload; every id must appear exactly once, which "
        "narrate.problems_with checks and a test enforces against the live trace.")


# Write the wording for one bundled puzzle. Returns the path written.
def regenerate(name, ask, out_dir=EXAMPLES):
    info, puzzle, clues = load(name)
    result = solve(puzzle, clues)
    if result.status is not Status.SOLVED:
        raise SystemExit(f"{name} does not solve ({result.status.value}); "
                         "fix the puzzle before narrating it.")

    groups = to_ai_payload(result.trace, clues)
    prose = narrate(groups, ask)

    # Belt and braces: narrate() already refuses anything that does not line up,
    # but this file is written once and read forever, so check it again here
    # against the very steps it will sit beside.
    problems = problems_with({str(k): v for k, v in prose.items()}, for_writing(groups))
    if problems:
        raise SystemExit("refusing to write wording that does not match the "
                         "proof: " + "; ".join(problems))

    path = pathlib.Path(out_dir) / f"{name}_prose.json"
    path.write_text(
        json.dumps({"_note": NOTE.format(title=info["name"], name=name),
                    "prose": {str(step): text for step, text in sorted(prose.items())}},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return path, len(prose)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-key", required=True, help="your own Anthropic API key")
    parser.add_argument("--name", help="which bundled puzzle to narrate")
    parser.add_argument("--all", action="store_true", help="narrate every bundled puzzle")
    args = parser.parse_args(argv)

    known = [example["name"] for example in available()]
    if args.all:
        names = known
    elif args.name:
        if args.name not in known:
            raise SystemExit(f"no bundled puzzle called '{args.name}'. "
                             f"Try one of: {', '.join(known)}")
        names = [args.name]
    else:
        raise SystemExit("say which puzzle: --name <one of "
                         f"{', '.join(known)}> or --all")

    ask = anthropic_asker(args.api_key)
    for name in names:
        print(f"narrating {name} ...", flush=True)
        try:
            path, count = regenerate(name, ask)
        except NarrationFailed as failed:
            last = "; ".join(failed.attempts[-1]) if failed.attempts else "no attempts"
            raise SystemExit(f"  gave up on {name}: {last}")
        print(f"  wrote {path} ({count} sentences)")

    print("\nNow run the tests - they check the new wording against the live "
          "trace, and that case-split steps still say they were settled by cases.")


if __name__ == "__main__":
    sys.exit(main())
