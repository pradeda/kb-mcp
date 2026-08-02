from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from kb_search_api import (
    NexusRelevanceRequest,
    SearchResult,
    kb_synthesize_nexus_relevance,
)


def request() -> NexusRelevanceRequest:
    return NexusRelevanceRequest(
        query="Nexus Fable 5 GPT 5.6 Sol OpenCode OpenRouter",
        video_title="Fable 5 vs GPT 5.6 Sol",
        video_summary="Comparison of model capability, cost and workflows.",
        initial_assessment="No direct Nexus relevance.",
        tools_models=["Fable 5", "GPT 5.6 Sol"],
    )


def result(entry_id: int, score: float) -> SearchResult:
    return SearchResult(
        id=entry_id,
        title=f"KB entry {entry_id}",
        content="x" * 3000,
        relevance=0.95,
        final_score=score,
    )


class NexusSynthesisTests(unittest.TestCase):
    def test_confirmed_response_keeps_only_retrieved_provenance(self) -> None:
        model_output = {
            "answer": "Directly relevant to the existing Nexus model workflow.",
            "kb_match_confirmed": True,
            "operational_relevance": "direct",
            "supporting_evidence": [
                {"entry_id": 400, "match_reason": "Same model family and API tiers."},
                {"entry_id": 401, "match_reason": "Same benchmark comparison."},
            ],
        }
        with patch(
            "kb_search_api._do_search",
            return_value=[result(400, 0.91), result(401, 0.82), result(999, 0.59)],
        ), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            response = kb_synthesize_nexus_relevance(
                request(), x_kb_synthesis_token="test"
            )

        self.assertEqual(response.status, "operationally_relevant")
        self.assertTrue(response.kb_match_confirmed)
        self.assertEqual(response.operational_relevance, "direct")
        self.assertEqual(
            [item.entry_id for item in response.supporting_entries], [400, 401]
        )
        self.assertEqual(
            response.supporting_entries[0].match_reason,
            "Same model family and API tiers.",
        )
        self.assertEqual(response.provenance.model_call_count, 1)
        context = provider.call_args.args[0]
        self.assertEqual(context["item"]["source_type"], "video")
        self.assertEqual([item["entry_id"] for item in context["kb_entries"]], [400, 401])
        self.assertTrue(all(len(item["excerpt"]) == 2000 for item in context["kb_entries"]))

    def test_article_source_type_reaches_synthesis_context(self) -> None:
        article_request = request().model_copy(update={"source_type": "article"})
        model_output = {
            "answer": "The article matches existing KB knowledge.",
            "kb_match_confirmed": True,
            "operational_relevance": "not_confirmed",
            "supporting_evidence": [{
                "entry_id": 400,
                "match_reason": "Both cover the same model release.",
            }],
        }
        with patch(
            "kb_search_api._do_search", return_value=[result(400, 0.91)]
        ), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            kb_synthesize_nexus_relevance(
                article_request, x_kb_synthesis_token="test"
            )
        self.assertEqual(provider.call_args.args[0]["item"]["source_type"], "article")

    def test_article_uses_lower_related_knowledge_candidate_threshold(self) -> None:
        article_request = request().model_copy(update={"source_type": "article"})
        model_output = {
            "answer": "The article has related KB knowledge but no operational impact.",
            "kb_match_confirmed": True,
            "operational_relevance": "not_confirmed",
            "supporting_evidence": [{
                "entry_id": 400,
                "match_reason": "Both discuss adoption of open-weight models.",
            }],
        }
        with patch(
            "kb_search_api._do_search", return_value=[result(400, 0.51)]
        ), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            response = kb_synthesize_nexus_relevance(
                article_request, x_kb_synthesis_token="test"
            )
        self.assertEqual(response.status, "kb_match_only")
        provider.assert_called_once()

    def test_model_cannot_cite_an_entry_not_returned_by_retrieval(self) -> None:
        model_output = {
            "answer": "Unsupported citation.",
            "kb_match_confirmed": True,
            "operational_relevance": "direct",
            "supporting_evidence": [
                {"entry_id": 777, "match_reason": "Unavailable source."},
            ],
        }
        with patch(
            "kb_search_api._do_search", return_value=[result(400, 0.91)]
        ), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ), patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            with self.assertRaises(HTTPException) as raised:
                kb_synthesize_nexus_relevance(
                    request(), x_kb_synthesis_token="test"
                )
        self.assertEqual(raised.exception.status_code, 502)

    def test_no_strong_match_skips_model_call(self) -> None:
        with patch(
            "kb_search_api._do_search", return_value=[result(400, 0.59)]
        ), patch(
            "kb_search_api._call_synthesis_model"
        ) as provider, patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            response = kb_synthesize_nexus_relevance(
                request(), x_kb_synthesis_token="test"
            )
        self.assertEqual(response.status, "not_confirmed")
        self.assertFalse(response.kb_match_confirmed)
        self.assertEqual(response.operational_relevance, "not_confirmed")
        self.assertFalse(response.connection_confirmed)
        self.assertEqual(response.provenance.model_call_count, 0)
        provider.assert_not_called()

    def test_topic_match_does_not_become_operational_relevance(self) -> None:
        model_output = {
            "answer": (
                "KB #400 corroborates the model-family topic. No concrete Nexus service, "
                "workflow, hardware or roadmap impact is confirmed."
            ),
            "kb_match_confirmed": True,
            "operational_relevance": "not_confirmed",
            "supporting_evidence": [{
                "entry_id": 400,
                "match_reason": "Both discuss the GPT-5.6 Sol, Terra and Luna model family.",
            }],
        }
        with patch(
            "kb_search_api._do_search", return_value=[result(400, 0.93)]
        ), patch(
            "kb_search_api._call_synthesis_model", return_value=model_output
        ), patch.dict("os.environ", {"KB_SYNTHESIS_TOKEN": "test"}):
            response = kb_synthesize_nexus_relevance(
                request(), x_kb_synthesis_token="test"
            )
        self.assertEqual(response.status, "kb_match_only")
        self.assertTrue(response.kb_match_confirmed)
        self.assertEqual(response.operational_relevance, "not_confirmed")
        self.assertFalse(response.connection_confirmed)
        self.assertEqual([item.entry_id for item in response.supporting_entries], [400])
        self.assertIn("Sol, Terra and Luna", response.supporting_entries[0].match_reason)

    def test_route_is_disabled_without_a_dedicated_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "kb_search_api._do_search"
        ) as retrieval:
            with self.assertRaises(HTTPException) as raised:
                kb_synthesize_nexus_relevance(
                    request(), x_kb_synthesis_token=None
                )
        self.assertEqual(raised.exception.status_code, 503)
        retrieval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
