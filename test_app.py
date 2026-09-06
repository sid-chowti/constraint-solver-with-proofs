from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


# The page itself has to load, or nothing else matters.
def test_the_page_loads():
    response = client.get("/")

    assert response.status_code == 200
    assert "Logic Puzzle Solver" in response.text


# The no-key path: the whole point of bundling a translated example.
def test_the_example_solves_without_a_key():
    body = client.get("/api/example").json()

    assert body["status"] == "solved"
    assert body["info"]["name"] == "The Einstein Puzzle"
    assert len(body["answer"]) == 5
    assert len(body["categories"]) == 5


# The answer must be the real one, not just any grid.
def test_the_example_gives_the_known_answer():
    rows = client.get("/api/example").json()["answer"]
    by_position = {row["position"]: row["values"] for row in rows}

    zebra = next(p for p, v in by_position.items() if v["pet"] == "zebra")
    water = next(p for p, v in by_position.items() if v["drink"] == "water")

    assert by_position[zebra]["nation"] == "Japanese"
    assert by_position[water]["nation"] == "Norwegian"


# The explanation is the product, so it has to come back with the answer.
def test_the_example_comes_with_its_explanation():
    body = client.get("/api/example").json()

    assert len(body["trace"]) > 50
    assert any(group["children"] for group in body["trace"])
    # Every group carries its own id, which is what makes the chain checkable.
    assert [g["id"] for g in body["trace"]] == list(range(1, len(body["trace"]) + 1))


# Each clue comes back three ways: the user's sentence, what the solver
# understood, and the data itself so the browser can hand it back.
def test_the_clues_come_back_both_ways():
    clues = client.get("/api/example").json()["clues"]

    assert len(clues) == 14
    assert clues[0]["source_text"] == "The Englishman lives in the red house."
    assert clues[0]["says"] == "English (nation) and red (color) are in the same house"
    assert clues[0]["also_means"] is None  # symmetric, so no second reading

    # The one direction-bearing clue in the whole puzzle is said both ways, so
    # the reader never has to flip it in their head.
    green = next(c for c in clues if "immediately" in c["says"])
    assert green["says"] == "ivory (color) is immediately left of green (color)"
    assert green["also_means"] == "green (color) is immediately right of ivory (color)"


# ---------------------------------------------------------------------------
# /api/translate - reads the puzzle, and deliberately stops there
# ---------------------------------------------------------------------------


def _fake_translation(monkeypatch):
    """Wire /api/translate to the bundled puzzle, so the AI never gets called."""
    import app as app_module
    from examples import load

    _, puzzle, clues = load()
    monkeypatch.setattr(app_module, "anthropic_asker", lambda key: None)
    monkeypatch.setattr(app_module, "translate", lambda text, ask, **kw: (puzzle, clues))
    return puzzle, clues


# The whole point of the split: translating must NOT solve. If an answer came
# back here there would be something to anchor on before the check.
def test_translating_does_not_solve(monkeypatch):
    _fake_translation(monkeypatch)

    body = client.post("/api/translate", json={"text": "a puzzle", "api_key": "sk-test"}).json()

    assert len(body["clues"]) == 14
    assert body["num_positions"] == 5
    for absent in ("status", "answer", "trace"):
        assert absent not in body


# A missing key is a clear message, not a crash or a server-side charge.
def test_translating_without_a_key_is_refused():
    response = client.post("/api/translate", json={"text": "a puzzle", "api_key": ""})

    assert response.status_code == 400
    assert "API key" in response.json()["detail"]


# An empty puzzle should not reach the API at all.
def test_an_empty_puzzle_is_refused():
    response = client.post("/api/translate", json={"text": "   ", "api_key": "sk-test"})

    assert response.status_code == 400


# A failure inside translation must surface as a clean error, never a stack
# trace, and never a 500.
def test_a_broken_translation_is_reported_cleanly(monkeypatch):
    import app as app_module
    from translate import TranslationFailed

    def fails(text, ask, **kwargs):
        raise TranslationFailed([["clue 1: unsupported operator"]])

    monkeypatch.setattr(app_module, "translate", fails)
    monkeypatch.setattr(app_module, "anthropic_asker", lambda key: None)

    response = client.post("/api/translate", json={"text": "a puzzle", "api_key": "sk-test"})

    assert response.status_code == 422
    assert "Could not read" in response.json()["detail"]["message"]
    assert "unsupported operator" in response.json()["detail"]["attempts"][0][0]


# If the API itself is unreachable or the key is bad, say so plainly.
def test_an_api_failure_is_reported_cleanly(monkeypatch):
    import app as app_module

    def explodes(key):
        raise RuntimeError("invalid x-api-key")

    monkeypatch.setattr(app_module, "anthropic_asker", explodes)

    response = client.post("/api/translate", json={"text": "a puzzle", "api_key": "bad"})

    assert response.status_code == 502
    assert "invalid x-api-key" in response.json()["detail"]


# Warnings ride down with the translation, so the person checking the reading
# sees them at the moment they can actually judge them.
def test_translate_carries_warnings(monkeypatch):
    _fake_translation(monkeypatch)

    body = client.post("/api/translate", json={"text": "x", "api_key": "k"}).json()

    assert body["warnings"] == []   # the Einstein puzzle has nothing odd in it


def test_a_value_in_two_categories_is_warned_about(monkeypatch):
    import app as app_module
    from parsing import clues_from_json
    from puzzle import Puzzle

    # "green" is both a colour and a pet — legal, but almost always confusion.
    puzzle = Puzzle({"color": ["red", "green"], "pet": ["cat", "green"]}, 2)
    clues = clues_from_json([{"type": "AbsolutePosition",
                              "category_value": ["color", "red"],
                              "operator": "==", "position": 1}])
    monkeypatch.setattr(app_module, "anthropic_asker", lambda key: None)
    monkeypatch.setattr(app_module, "translate", lambda text, ask, **kw: (puzzle, clues))

    body = client.post("/api/translate", json={"text": "x", "api_key": "k"}).json()

    assert len(body["warnings"]) == 1
    assert "green" in body["warnings"][0]
    # A warning must not block: the clues still came through.
    assert len(body["clues"]) == 1


# ---------------------------------------------------------------------------
# /api/solve - the confirmed clues come back up and get solved
# ---------------------------------------------------------------------------


def _confirm(body, keep=None):
    """Turn a /api/translate reply into a /api/solve request, the way the page
    does. `keep` picks clue numbers, mimicking the checkboxes."""
    rows = body["clues"] if keep is None else [c for c in body["clues"] if c["number"] in keep]
    return {"num_positions": body["num_positions"],
            "categories": body["categories"],
            "clues": [row["clue"] for row in rows]}


# The round trip: whatever translate hands down must be solvable when handed
# straight back up. This is the join between the two requests, and the server
# remembers nothing in between.
def test_the_confirmation_round_trip_solves(monkeypatch):
    _fake_translation(monkeypatch)
    translated = client.post("/api/translate", json={"text": "x", "api_key": "k"}).json()

    body = client.post("/api/solve", json=_confirm(translated)).json()

    assert body["status"] == "solved"
    by_position = {row["position"]: row["values"] for row in body["answer"]}
    zebra = next(p for p, v in by_position.items() if v["pet"] == "zebra")
    assert by_position[zebra]["nation"] == "Japanese"
    assert body["trace"]


# Solving is pure logic - no AI, so no key, so no cost. Nothing in this
# endpoint should ever want one.
def test_solving_needs_no_api_key(monkeypatch):
    _fake_translation(monkeypatch)
    translated = client.post("/api/translate", json={"text": "x", "api_key": "k"}).json()

    assert client.post("/api/solve", json=_confirm(translated)).status_code == 200


# Switching a clue off can only ever WIDEN the set of valid answers, so the
# result is either the same answer or an honest INCOMPLETE - never a wrong
# answer. That property is what makes the checkboxes safe.
def test_switching_a_clue_off_never_gives_a_wrong_answer(monkeypatch):
    _fake_translation(monkeypatch)
    translated = client.post("/api/translate", json={"text": "x", "api_key": "k"}).json()
    full = client.post("/api/solve", json=_confirm(translated)).json()["answer"]

    for dropped in (5, 8, 9):  # a direction clue, and both house-number clues
        keep = [n for n in range(1, 15) if n != dropped]
        body = client.post("/api/solve", json=_confirm(translated, keep)).json()

        assert body["status"] in ("solved", "incomplete")
        if body["status"] == "solved":
            assert body["answer"] == full


# Every clue in the Einstein puzzle turns out to be load-bearing, so the test
# above only ever sees the "incomplete" half of the property. This is the other
# half: a puzzle with a clue that adds nothing, which must still solve to the
# very same answer once that clue is switched off.
def test_switching_off_a_redundant_clue_keeps_the_same_answer():
    puzzle = {"num_positions": 3,
              "categories": {"color": ["red", "green", "blue"],
                             "pet": ["dog", "cat", "fox"]}}
    needed = [
        {"type": "AbsolutePosition", "category_value": ["color", "red"],
         "operator": "==", "position": 1},
        {"type": "AbsolutePosition", "category_value": ["color", "green"],
         "operator": "==", "position": 2},
        {"type": "AbsolutePosition", "category_value": ["pet", "dog"],
         "operator": "==", "position": 1},
        {"type": "AbsolutePosition", "category_value": ["pet", "cat"],
         "operator": "==", "position": 2},
    ]
    # True, but already forced by the four above: blue and fox have nowhere
    # else left to go.
    redundant = {"type": "AbsolutePosition", "category_value": ["color", "blue"],
                 "operator": "==", "position": 3}

    with_it = client.post("/api/solve", json={**puzzle, "clues": needed + [redundant]}).json()
    without_it = client.post("/api/solve", json={**puzzle, "clues": needed}).json()

    assert with_it["status"] == "solved"
    assert without_it["status"] == "solved"
    assert without_it["answer"] == with_it["answer"]


# Switching every clue off is not a crash, just a puzzle nothing is known about.
def test_switching_every_clue_off_is_honest(monkeypatch):
    _fake_translation(monkeypatch)
    translated = client.post("/api/translate", json={"text": "x", "api_key": "k"}).json()

    body = client.post("/api/solve", json=_confirm(translated, keep=[])).json()

    assert body["status"] == "incomplete"
    assert body["answer"] is None


# ---------------------------------------------------------------------------
# /api/solve trusts nothing it is sent - the browser is no more trusted than
# the AI was, and both meet the same guards.
# ---------------------------------------------------------------------------


def test_clues_that_are_not_clues_are_refused():
    response = client.post("/api/solve", json={
        "num_positions": 2,
        "categories": {"color": ["red", "blue"]},
        "clues": [{"type": "Nonsense", "operator": "??"}],
    })

    assert response.status_code == 400
    assert response.json()["detail"]["problems"]


def test_a_clue_naming_something_unknown_is_refused():
    response = client.post("/api/solve", json={
        "num_positions": 2,
        "categories": {"color": ["red", "blue"]},
        "clues": [{"type": "AbsolutePosition", "category_value": ["color", "purple"],
                   "operator": "==", "position": 1}],
    })

    assert response.status_code == 400
    assert "purple" in str(response.json()["detail"])


def test_a_puzzle_of_the_wrong_shape_is_refused():
    response = client.post("/api/solve", json={
        "num_positions": 5,
        "categories": {"color": ["red", "blue"]},  # 2 values for 5 houses
        "clues": [],
    })

    assert response.status_code == 400
    assert "not a usable puzzle" in response.json()["detail"]


# The page is plain JavaScript, so nothing else here would notice it calling an
# endpoint that no longer exists. This is the cheap guard against that drift:
# it would have caught the moment the endpoints split and the page did not.
def test_the_page_calls_the_endpoints_that_exist():
    page = client.get("/").text
    routes = {r.path for r in app.routes}

    for called in ("/api/translate", "/api/solve", "/api/example"):
        assert called in page, f"the page never calls {called}"
        assert called in routes, f"{called} is not a route"

    # /api/solve must be sent the puzzle and its clues - never English and a
    # key, which is what the old combined endpoint took.
    assert 'num_positions: translated.num_positions' in page


# The confirmation screen has to exist in the markup for the flow to work at
# all, and its two buttons are what the script hangs its handlers on.
def test_the_confirmation_screen_is_on_the_page():
    page = client.get("/").text

    for piece in ('id="confirm"', 'id="confirm-clues"', 'id="run-confirmed"',
                  'id="back-to-text"', "Check what it understood"):
        assert piece in page, f"missing {piece}"


# ---------------------------------------------------------------------------
# /api/narrate — the wording layer, opt-in and separately paid for
# ---------------------------------------------------------------------------


# The bundled puzzle ships already written, which is what keeps the demo free.
def test_the_example_comes_with_its_wording():
    body = client.get("/api/example").json()

    assert body["prose"], "the bundled puzzle should ship narrated"
    assert len(body["prose"]) == len(body["trace"])
    assert set(body["prose"]) == {str(g["id"]) for g in body["trace"]}


def _solvable():
    from examples import load

    _, puzzle, clues = load()
    from parsing import clue_to_json
    return {"num_positions": puzzle.num_positions,
            "categories": puzzle.categories,
            "clues": [clue_to_json(c) for c in clues]}


def test_narrating_without_a_key_is_refused():
    response = client.post("/api/narrate", json={**_solvable(), "api_key": "  "})

    assert response.status_code == 400
    assert "API key" in response.json()["detail"]


def test_narrating_returns_a_sentence_for_every_step(monkeypatch):
    import json

    import app as app_module

    def fake_asker(key):
        def ask(conversation):
            # Echo back one sentence per id the request actually asked about.
            asked = json.loads(conversation[0]["content"].split("\n\n", 1)[1])
            return json.dumps({str(s["id"]): f"Step {s['id']} in words." for s in asked})
        return ask

    monkeypatch.setattr(app_module, "narrating_asker", fake_asker)

    body = client.post("/api/narrate", json={**_solvable(), "api_key": "k"}).json()

    assert len(body["prose"]) == 74
    assert body["prose"]["1"] == "Step 1 in words."


# Half-narrated is worse than not narrated, so a reply that does not line up is
# thrown away whole and reported.
def test_wording_that_does_not_line_up_is_thrown_away(monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "narrating_asker",
                        lambda key: (lambda conversation: '{"1": "only this one"}'))

    response = client.post("/api/narrate", json={**_solvable(), "api_key": "k"})

    assert response.status_code == 422
    assert "thrown away" in response.json()["detail"]["message"]
    assert response.json()["detail"]["problems"]


def test_a_writing_api_failure_is_reported_cleanly(monkeypatch):
    import app as app_module

    def explodes(key):
        raise RuntimeError("invalid x-api-key")

    monkeypatch.setattr(app_module, "narrating_asker", explodes)

    response = client.post("/api/narrate", json={**_solvable(), "api_key": "bad"})

    assert response.status_code == 502
    assert "invalid x-api-key" in response.json()["detail"]


# Narrating runs the same guards as solving: it is handed data by a browser.
def test_narrating_forged_clues_is_refused():
    response = client.post("/api/narrate", json={
        "num_positions": 2,
        "categories": {"color": ["red", "blue"]},
        "clues": [{"type": "Bogus"}],
        "api_key": "k",
    })

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Choosing between the bundled puzzles
# ---------------------------------------------------------------------------


def test_the_bundled_puzzles_are_listed():
    body = client.get("/api/examples").json()

    names = [example["name"] for example in body["examples"]]
    assert "einstein" in names and len(names) >= 3
    # The famous one leads, because it is the reason anyone is here.
    assert names[0] == "einstein"
    for example in body["examples"]:
        assert example["title"] and example["question"]
        assert example["positions"] >= 2 and example["clues"] >= 1


def test_each_bundled_puzzle_can_be_solved_by_name():
    for example in client.get("/api/examples").json()["examples"]:
        body = client.get(f"/api/example?name={example['name']}").json()

        assert body["status"] == "solved", example["name"]
        assert len(body["answer"]) == example["positions"]
        assert body["trace"]


# The name arrives from a query string, so it must never reach the filesystem
# unchecked - otherwise "../../something" would be a valid example name.
def test_an_unknown_example_is_a_clean_404():
    assert client.get("/api/example?name=nope").status_code == 404


def test_an_example_name_cannot_escape_the_examples_folder():
    for attack in ("../app", "../../etc/passwd", "einstein/../../app"):
        response = client.get("/api/example", params={"name": attack})

        assert response.status_code == 404, attack
