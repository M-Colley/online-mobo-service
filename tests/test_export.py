"""Offline export regressions: versioned questionnaires and malformed scores."""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import space
from analysis.export_firestore import COLUMNS, LONG_COLUMNS, build_rows


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.proposal = {
            "pid": "001", "phaseStep": 1, "schemaVersion": space.SCHEMA_VERSION,
            "mobo": {"phase": "anchor", "spaceVersion": space.SPACE_VERSION},
            "haptics": space.expand(space.anchor()),
        }
        self.current = {
            "pid": "001", "phaseStep": 1, "roundNumber": 1,
            "phase": "exploration", "schemaVersion": space.SCHEMA_VERSION,
            "parameterDocumentId": "candidate-one", "candidateId": "one",
            "sessionId": "session-one", "resultId": "session-one-round-1",
            "haptics": self.proposal["haptics"],
            "subjectiveScore": 0.75, "objectiveScore": 0.4,
            "timeToTargetSeconds": 56.0, "touchedTarget": True,
            "questionnaireVersion": space.QUESTIONNAIRE_VERSION,
            "q_cognitive_load": 6, "q_clear": 4, "q_likability": 5,
            "q_comfort": 3,
        }

    def export(self, docs):
        with contextlib.redirect_stdout(io.StringIO()):
            return build_rows(docs, {"candidate-one": self.proposal},
                              {("001", 1): self.proposal}, {("001", 1): 0.25})

    def legacy(self):
        result = copy.deepcopy(self.current)
        for field in ("questionnaireVersion", "q_cognitive_load", "q_likability"):
            result.pop(field)
        result.update(resultId="legacy-round-1", q_pleasant=5, q_distinct=4,
                      q_clear=6, q_comfort=7)
        return result

    def test_current_raw_answers_and_identity_survive_both_exports(self):
        rows, long_rows, skipped, _ = self.export([("doc-one", self.current)])
        self.assertEqual(skipped, 0)
        row = rows[0]
        self.assertTrue(row["trainingEligible"])
        self.assertEqual(row["pid"], "001")
        for field in ("resultId", "schemaVersion", "questionnaireVersion",
                      "q_cognitive_load", "q_clear", "q_likability", "q_comfort",
                      "timeToTargetSeconds"):
            self.assertEqual(row[field], self.current[field])
        self.assertEqual(set(row), set(COLUMNS))
        self.assertEqual(len(long_rows), len(space.CUES))
        for cue in long_rows:
            self.assertEqual(set(cue), set(LONG_COLUMNS))
            self.assertEqual(cue["resultId"], row["resultId"])
            self.assertEqual(cue["sessionId"], row["sessionId"])
            self.assertTrue(cue["trainingEligible"])

    def test_legacy_scores_retained_without_inventing_new_responses(self):
        legacy = self.legacy()
        rows, _, _, _ = self.export([("legacy", legacy)])
        row = rows[0]
        self.assertFalse(row["trainingEligible"])
        self.assertEqual(row["hypervolume"], "")
        self.assertEqual(row["questionnaireVersion"], "")
        self.assertEqual(row["q_cognitive_load"], "")
        self.assertEqual(row["q_likability"], "")
        for field in ("q_pleasant", "q_distinct", "q_clear", "q_comfort",
                      "subjectiveScore", "objectiveScore"):
            self.assertEqual(row[field], legacy[field])

    def test_legacy_round_does_not_shadow_current_questionnaire(self):
        rows, _, skipped, _ = self.export([
            ("legacy", self.legacy()), ("current", self.current)])
        self.assertEqual(len(rows), 2)
        self.assertEqual(skipped, 0)
        self.assertEqual([r["trainingEligible"] for r in rows], [False, True])

    def test_invalid_current_questionnaire_does_not_shadow_valid_round(self):
        invalid = {**self.current, "q_cognitive_load": 0, "resultId": "invalid"}
        rows, _, skipped, _ = self.export([
            ("invalid", invalid), ("current", self.current)])
        self.assertEqual(len(rows), 2)
        self.assertEqual(skipped, 0)
        self.assertEqual([r["trainingEligible"] for r in rows], [False, True])

    def test_corrupt_scores_are_skipped_before_claiming_the_round(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5", None):
            with self.subTest(value=value):
                invalid = {**self.current, "objectiveScore": value}
                rows, _, skipped, _ = self.export([
                    ("invalid", invalid), ("current", self.current)])
                self.assertEqual(len(rows), 1)
                self.assertEqual(skipped, 1)
                self.assertTrue(rows[0]["trainingEligible"])

    def test_rounds_outside_app_budget_are_not_exported(self):
        for step in (0, -1, space.MAX_ROUND_NUMBER + 1):
            with self.subTest(step=step):
                rows, _, skipped, _ = self.export([
                    ("bad-step", {**self.current, "phaseStep": step})])
                self.assertEqual(rows, [])
                self.assertEqual(skipped, 1)

    def test_legacy_proposal_schema_cannot_shadow_the_current_fallback(self):
        legacy_proposal = {**self.proposal, "schemaVersion": 1,
                           "mobo": {"spaceVersion": "old-space"}}
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _, skipped, _ = build_rows(
                [("current", self.current)], {"candidate-one": legacy_proposal},
                {("001", 1): self.proposal}, {})
        self.assertEqual(skipped, 0)
        self.assertTrue(rows[0]["trainingEligible"])
        self.assertEqual(rows[0]["optimizerPhase"], "anchor")

    def test_foreign_participant_proposal_cannot_change_round_metadata(self):
        foreign = {**self.proposal, "pid": "another-participant",
                   "mobo": {"spaceVersion": "foreign-space"}}
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _, skipped, _ = build_rows(
                [("current", self.current)], {"candidate-one": foreign},
                {("001", 1): self.proposal}, {})
        self.assertEqual(skipped, 0)
        self.assertEqual(rows[0]["optimizerPhase"], "anchor")

    def test_failed_attention_round_keeps_its_own_haptic_join_identity(self):
        failed = {**self.current, "attentionCheckPassed": False,
                  "resultId": "failed-round-1"}
        rows, long_rows, _, _ = self.export([
            ("failed", failed), ("current", self.current)])
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["trainingEligible"] for r in rows], [False, True])
        self.assertEqual({r["resultId"] for r in long_rows},
                         {"failed-round-1", "session-one-round-1"})


if __name__ == "__main__":
    unittest.main()
