"""Did the AI invent a value that the user never wrote?

The third guard. The first two check the AI against itself — that its clues are
readable, and that they name things its own puzzle defines. Neither notices a
value the AI made up out of nothing, because a hallucinated value is perfectly
consistent with the rest of a hallucinated puzzle. This one checks the AI
against the USER: every value it lists should be traceable to the words they
actually typed.

Deliberately one-sided. It only ever asks "does this word appear?" — plain text
search, no model, no meaning. It cannot tell whether a value was understood
correctly, only whether it was pulled out of thin air.

LEAN PERMISSIVE. A false alarm blocks somebody whose puzzle was fine, which is
much worse than letting an invented value through — the confirmation screen
shows every value to the user anyway, so a miss here is caught by a person, but
a false alarm stops them ever getting to that screen. The rules below are all
consequences of that: matching the real Einstein puzzle needs every one of them.

    value "English"     text says "the Englishman lives..."   prefix of a word
    value "OldGold"     text says "the Old Gold smoker..."    the space is gone
    value "Kools"       text says "Kool is smoked..."         a plural twin

Category NAMES are never checked. People write "the red house", not "colour".
"""
import re

# "LuckyStrike" -> ["Lucky", "Strike"]. Also splits on spaces, hyphens and
# underscores, so every spelling of a two-word value ends up as the same parts.
_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+")


# Break a value into the words it is made of, however it was spelled.
def _words(value):
    return _WORDS.findall(value)


# Turn a value into a regex that matches the way a person would have written it.
def _pattern(value):
    words = _words(value)
    if not words:
        return None

    # Words may be run together, spaced, or hyphenated in the user's text.
    joined = r"[\s\-_]*".join(re.escape(word) for word in words)

    # A leading word boundary, but deliberately NO trailing one: "English" has
    # to match "Englishman", and "Kool" has to match "Kools". Dropping the
    # trailing boundary is what makes this permissive rather than pedantic.
    return re.compile(r"\b" + joined, re.IGNORECASE)


# Every spelling of a value worth trying: the value itself, then its singular.
def _candidates(value):
    yield value
    lowered = value.lower()
    if lowered.endswith("es") and len(value) > 3:
        yield value[:-2]
    if lowered.endswith("s") and len(value) > 2:
        yield value[:-1]


# True if `value` can be traced to something the user actually wrote.
def is_grounded(value, text):
    for candidate in _candidates(value):
        pattern = _pattern(candidate)
        if pattern is not None and pattern.search(text):
            return True
    return False


# Every (category, value) the AI listed that appears nowhere in the user's text.
def find_ungrounded(categories, text):
    missing = []
    for category, values in categories.items():
        for value in values:
            if not is_grounded(value, text):
                missing.append((category, value))
    return missing


# The complaints to hand back to the AI, in the same shape the other guards use.
def complaints(categories, text):
    return [
        f"the {category} value '{value}' does not appear anywhere in the "
        f"puzzle text — use the words the puzzle actually uses, and never "
        f"invent a value"
        for category, value in find_ungrounded(categories, text)
    ]
