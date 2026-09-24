import re

# Common generic tech-product suffixes that are frequently invented and
# attached to a real company name to sound plausible (e.g. "Meta Cloud",
# "Facebook Cloud") — this targets the exact failure pattern observed when
# testing the model's output against real data.
SUSPICIOUS_SUFFIXES = (
    "Cloud", "AI", "Copilot", "Assistant", "Suite", "Platform", "Hub",
    "Studio", "Drive", "Docs", "Chat", "Bot", "Engine",
)

PRODUCT_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]+)\s+(" + "|".join(SUSPICIOUS_SUFFIXES) + r")\b"
)

# Common sentence-starters/generic words that capitalize at the start of a
# sentence or clause and can precede "AI" etc. incidentally (e.g. "As AI
# continues to..."), not as part of a product name.
NOT_A_COMPANY_NAME = {
    "as", "the", "this", "that", "these", "those", "many", "some", "most",
    "other", "such", "new", "more", "less", "all", "any", "each", "every",
    "its", "our", "their", "his", "her", "in", "on", "for", "with", "and",
    "or", "but", "increasing", "growing", "strong", "key", "major", "while",
    "given", "since", "when", "where", "which", "who", "if", "because",
    "expanding", "continuing", "driving", "leveraging", "investing",
    "building", "improving", "strengthening", "reducing", "boosting",
    "advancing", "accelerating", "enhancing", "supporting",
}

# A bare 4-digit number that looks like a calendar year (e.g. from a
# self-generated date header) is a date reference, not a data citation.
_YEAR_RANGE = range(2000, 2100)

# Real products that would otherwise get flagged just for not appearing in
# a given ticker's fetched data (e.g. "Google Cloud" is real even if this
# run's news/fundamentals never mention it).
KNOWN_REAL_PRODUCTS = {
    "amazon web services", "aws", "google cloud", "azure", "microsoft azure",
    "azure ai", "google ai", "microsoft ai", "office 365", "microsoft teams",
    "dynamics 365", "microsoft graph", "google workspace", "icloud",
    "apple music", "google drive", "google docs", "chatgpt", "openai",
    "github copilot", "microsoft copilot", "copilot studio", "oracle cloud",
    "ibm cloud", "salesforce",
}

# Numbers with a decimal point (digits on both sides), a %, or >= 10 are
# treated as "specific figures" worth checking; small bare integers and
# trailing list-numbering periods ("2.", "3.") are not.
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

# Numbers that are part of a named index/phrase ("S&P 500", "Nasdaq 100",
# "52-week high") rather than a cited data figure — stripped out before
# number-matching so they're never flagged.
NAMED_NUMBER_PHRASES = re.compile(
    r"\b(S&P|Dow(?:\s+Jones)?|Nasdaq|Russell|Fortune)\s+\d[\d,]*\b"
    r"|\b\d{1,3}-(?:week|day|year)\b",
    re.IGNORECASE,
)


def _flatten_numbers(data) -> set:
    numbers = set()

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, bool):
            return
        elif isinstance(obj, (int, float)):
            numbers.add(round(float(obj), 2))

    walk(data)
    return numbers


def _is_citable(raw: str) -> bool:
    stripped = raw.rstrip("%")
    if not stripped or stripped == "-":
        return False
    try:
        value = float(stripped.replace(",", ""))
    except ValueError:
        return False
    if not raw.endswith("%") and "." not in stripped and int(value) in _YEAR_RANGE:
        return False  # looks like a calendar year, not a data citation
    return raw.endswith("%") or "." in stripped or abs(value) >= 10


def check_numbers(analysis_text: str, source_data: dict, tolerance: float = 0.75) -> list:
    """Return figures mentioned in analysis_text that don't approximately
    match any number present in source_data."""
    reference_numbers = _flatten_numbers(source_data)
    cleaned_text = NAMED_NUMBER_PHRASES.sub("", analysis_text)

    flagged = []
    seen = set()
    for match in NUMBER_PATTERN.finditer(cleaned_text):
        raw = match.group()
        if raw in seen or not _is_citable(raw):
            continue
        seen.add(raw)

        value = float(raw.rstrip("%").replace(",", ""))

        # A "%"-suffixed claim may correspond to a reference value stored as
        # a raw decimal fraction (e.g. roe: 0.3969 -> claimed as "39.69%"),
        # so check both the value itself and value/100 against the data.
        candidates = (value, value / 100) if raw.endswith("%") else (value,)
        if not any(
            abs(candidate - ref) <= tolerance
            for candidate in candidates
            for ref in reference_numbers
        ):
            flagged.append(raw)

    return flagged


def check_named_products(analysis_text: str, source_summary: str) -> list:
    """Return "{Company} {Suffix}"-style product names mentioned in
    analysis_text (e.g. "Meta Cloud") that aren't a known real product and
    don't appear verbatim in the real source data."""
    source_lower = source_summary.lower()

    flagged = []
    seen = set()
    for match in PRODUCT_PATTERN.finditer(analysis_text):
        phrase = match.group()
        if phrase in seen:
            continue
        seen.add(phrase)

        if match.group(1).lower() in NOT_A_COMPANY_NAME:
            continue

        normalized = phrase.lower()
        if normalized in KNOWN_REAL_PRODUCTS or normalized in source_lower:
            continue
        flagged.append(phrase)

    return flagged


def check_grounding(analysis_text: str, source_data: dict, source_summary: str) -> dict:
    """Deterministic, rule-based grounding check — no second model call.
    Catches the two concrete failure modes observed in testing: fabricated
    figures and invented "{Company} {Suffix}" product names. It will not
    catch subtler issues (e.g. a real number mischaracterized as "high"
    when it's actually low) — that's a generation-quality problem, not
    something a post-hoc check can fix."""
    unverified_numbers = check_numbers(analysis_text, source_data)
    unverified_products = check_named_products(analysis_text, source_summary)

    return {
        "clean": not unverified_numbers and not unverified_products,
        "unverified_numbers": unverified_numbers,
        "unverified_products": unverified_products,
    }
