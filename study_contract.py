"""Validation shared by the service and the read-only data exporter."""

import math

import space

QUESTIONNAIRE_VERSION = space.QUESTIONNAIRE_VERSION
QUESTIONNAIRE_FIELDS = ("q_cognitive_load", "q_clear", "q_likability", "q_comfort")


def validate_scores(result: dict) -> list[float]:
    """Return finite 0–1 objectives; strings and booleans are not scores."""
    scores = []
    for field in space.OBJECTIVE_FIELDS:
        value = result.get(field)
        if (type(value) not in (int, float) or not 0 <= value <= 1
                or not math.isfinite(value)):
            raise ValueError(f"{field} must be a finite number in [0, 1]")
        scores.append(float(value))
    return scores


def validate_questionnaire(result: dict) -> None:
    """Require the current app's raw answers and matching normalized score.

    Haptic schemaVersion and questionnaireVersion describe separate contracts.
    Legacy scores are retained in exports, but cannot train the current model.
    """
    version = result.get("questionnaireVersion")
    if type(version) is not int or version != QUESTIONNAIRE_VERSION:
        raise ValueError(f"questionnaireVersion must be {QUESTIONNAIRE_VERSION}")
    for field in QUESTIONNAIRE_FIELDS:
        value = result.get(field)
        upper = 21 if field == "q_cognitive_load" else 5
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{field} must be an integer in [1, {upper}]")
    expected = ((21 - result["q_cognitive_load"]) / 20
                + sum((result[field] - 1) / 4 for field in QUESTIONNAIRE_FIELDS[1:])) / 4
    validate_scores(result)
    if not math.isclose(result["subjectiveScore"], expected, rel_tol=0, abs_tol=1e-9):
        raise ValueError("subjectiveScore does not match questionnaireVersion 2 answers")
