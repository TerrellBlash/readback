"""
Field-by-field scoring for the evaluation harness.

Each field has a kind that says how to compare it, so formatting
differences don't count as errors and value differences always do:
dates in any common printed format, amounts within a cent, company names
without suffixes like LLC, phone numbers by digits.

Free-text fields (summaries, contract terms) are scored on presence only.
Judging their wording would need a second model as a judge, and its
errors would end up in these numbers.

Every field gets one outcome:
- correct: matches, or the document lacks it and the extraction is empty
- wrong: a value that doesn't match
- missed: empty, though the document has it
- filled_absent: a value, even a placeholder like "N/A", for a field the
  document lacks. The confidence gate counts any value as present, so a
  placeholder slips past it just like an invented value.
- not_extracted: the pipeline produced no extraction for this document
"""

import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Literal

Outcome = Literal["correct", "wrong", "missed", "filled_absent", "not_extracted"]

FIELD_KINDS: dict[str, dict[str, str]] = {
    "invoice": {
        "vendor_name": "company",
        "invoice_number": "id",
        "invoice_date": "date",
        "due_date": "date",
        "line_items": "line_items",
        "subtotal": "amount",
        "tax": "amount",
        "total_amount": "amount",
        "currency": "currency",
    },
    "resume": {
        "full_name": "text",
        "email": "email",
        "phone": "phone",
        "location": "place",
        "summary": "presence",
        "skills": "skills",
        "experience": "names:company",
        "education": "names:institution",
    },
    "contract": {
        "contract_type": "contract_type",
        "parties": "names:name",
        "effective_date": "date",
        "expiration_date": "date",
        "key_terms": "presence",
        "payment_terms": "presence",
        "termination_clause": "presence",
        "governing_law": "place",
    },
}

_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %b %Y", "%d %B %Y", "%m/%d/%Y")
_COMPANY_SUFFIXES = {"inc", "llc", "ltd", "co", "corp", "corporation", "company", "gmbh", "plc", "lp", "llp"}
_CURRENCY_SYMBOLS = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP"}
_CONTRACT_TYPE_KEYWORDS = {
    "Master Services Agreement": ("master service", "msa"),
    "Consulting Agreement": ("consult",),
    "Software License Agreement": ("license", "licence"),
    "Commercial Lease Agreement": ("lease",),
    "Supply Agreement": ("supply",),
}
SKILLS_MIN_F1 = 0.8


def score_field(kind: str, expected: Any, actual: Any) -> Outcome:
    """
    Compare one extracted field with the ground truth.

    - Parameter kind: how to compare, from FIELD_KINDS. "names:<key>" means
        a list whose items are compared by that key.
    - Parameter expected: ground truth; None when the document lacks the field.
    - Parameter actual: what the pipeline extracted.
    - Returns: the field's outcome (see the module docstring).
    """
    if _is_empty(expected):
        return "correct" if _is_empty(actual) else "filled_absent"
    if _is_empty(actual):
        return "missed"
    base, _, key = kind.partition(":")
    try:
        matched = _MATCHERS[base](expected, actual, key)
    except (TypeError, ValueError, AttributeError):
        matched = False
    return "correct" if matched else "wrong"


def score_document(
    doc_type: str, expected_fields: dict[str, Any], extracted_fields: dict[str, Any] | None
) -> dict[str, Outcome]:
    """
    Score every ground-truth field of one document.

    - Parameter doc_type: the document's true type, which picks the field kinds.
    - Parameter expected_fields: ground truth, field name to value (None if absent).
    - Parameter extracted_fields: the pipeline's fields, or None if it
        produced no extraction (misclassified, or stopped before extracting).
    - Returns: field name to outcome.
    """
    if extracted_fields is None:
        return {name: "not_extracted" for name in expected_fields}
    kinds = FIELD_KINDS[doc_type]
    return {
        name: score_field(kinds[name], expected, extracted_fields.get(name))
        for name, expected in expected_fields.items()
    }


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


def _norm(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip().casefold()
    return text.strip(" .,;:")


def _company(value: Any) -> str:
    words = re.sub(r"[.,]", " ", _norm(value).replace("&", " and ")).split()
    while words and words[-1] in _COMPANY_SUFFIXES:
        words.pop()
    return " ".join(words)


def _date(value: Any) -> date:
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", str(value).strip()).replace(".", "")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date: {value!r}")


def _amount(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(re.sub(r"[^\d.\-]", "", str(value)))


def _currency(value: Any) -> str:
    text = str(value).strip()
    return _CURRENCY_SYMBOLS.get(text, text).upper()


def _match_text(expected: Any, actual: Any, _: str) -> bool:
    return _norm(expected) == _norm(actual)


def _match_id(expected: Any, actual: Any, _: str) -> bool:
    def clean(value: Any) -> str:
        return re.sub(r"\s+", "", _norm(value)).lstrip("#")

    return clean(expected) == clean(actual)


def _match_company(expected: Any, actual: Any, _: str) -> bool:
    return _company(expected) == _company(actual)


def _match_date(expected: Any, actual: Any, _: str) -> bool:
    return _date(expected) == _date(actual)


def _match_amount(expected: Any, actual: Any, _: str) -> bool:
    return abs(_amount(expected) - _amount(actual)) <= 0.01


def _match_currency(expected: Any, actual: Any, _: str) -> bool:
    return _currency(expected) == _currency(actual)


def _match_phone(expected: Any, actual: Any, _: str) -> bool:
    def digits(value: Any) -> str:
        return re.sub(r"\D", "", str(value))[-10:]

    return digits(expected) == digits(actual)


def _match_place(expected: Any, actual: Any, _: str) -> bool:
    return re.search(rf"\b{re.escape(_norm(expected))}\b", _norm(actual)) is not None


def _match_presence(expected: Any, actual: Any, _: str) -> bool:
    return True  # both non-empty; wording isn't judged


def _match_skills(expected: Any, actual: Any, _: str) -> bool:
    want, got = {_norm(s) for s in expected}, {_norm(s) for s in actual}
    overlap = len(want & got)
    if not overlap:
        return False
    precision, recall = overlap / len(got), overlap / len(want)
    return 2 * precision * recall / (precision + recall) >= SKILLS_MIN_F1


def _match_line_items(expected: Any, actual: Any, _: str) -> bool:
    if len(expected) != len(actual):
        return False
    want = sorted(_amount(item["total"]) for item in expected)
    got = sorted(_amount(item["total"]) for item in actual)
    return all(abs(a - b) <= 0.01 for a, b in zip(want, got, strict=False))


def _match_names(expected: Any, actual: Any, key: str) -> bool:
    # An entry without a name (a job with no employer) adds nothing to
    # compare; str(None) would otherwise make it a company called "none".
    names = (item.get(key) if isinstance(item, dict) else item for item in actual)
    got = {_company(name) for name in names if not _is_empty(name)}
    return {_company(name) for name in expected} == got


def _match_contract_type(expected: Any, actual: Any, _: str) -> bool:
    keywords = _CONTRACT_TYPE_KEYWORDS.get(expected, (_norm(expected),))
    return any(keyword in _norm(actual) for keyword in keywords)


_MATCHERS: dict[str, Callable[[Any, Any, str], bool]] = {
    "text": _match_text,
    "id": _match_id,
    "email": _match_text,
    "company": _match_company,
    "date": _match_date,
    "amount": _match_amount,
    "currency": _match_currency,
    "phone": _match_phone,
    "place": _match_place,
    "presence": _match_presence,
    "skills": _match_skills,
    "line_items": _match_line_items,
    "names": _match_names,
    "contract_type": _match_contract_type,
}
