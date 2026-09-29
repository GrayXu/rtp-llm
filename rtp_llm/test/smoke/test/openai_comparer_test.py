import copy
import json
import unittest
from pathlib import Path

from smoke.common_def import QueryStatus, SmokeException, Tracer
from smoke.openai_comparer import OpenaiComparer

GRAPH_STATUS = "generation_prefill_cuda_graph_status"
REPLAYED = "replayed"
FALLBACK = "input_tokens_exceed_capture_limit"


class OpenaiComparerGraphStatusTest(unittest.TestCase):
    def setUp(self):
        # The OpenAI graph smoke goldens require the status via compare_config,
        # without specifying result.aux_info (which contains runtime metrics).
        self.golden = {
            "id": "chat-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "same output"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        qr_info = {
            "query": {
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
            "result": self.golden,
            "compare_config": {"required_aux_info": {GRAPH_STATUS: REPLAYED}},
        }
        self.comparer = OpenaiComparer(None, "", qr_info, Tracer(), False)
        self.query = self.comparer.format_query(qr_info["query"])

    def _response(self, aux_info):
        payload = copy.deepcopy(self.golden)
        payload["aux_info"] = aux_info
        return payload

    def _parse(self, payload):
        return self.comparer.format_result(
            self.comparer.curl_response_to_json(self.query, json.dumps(payload))
        )

    def _compare(self, expected_status, actual_payload):
        self.comparer.qr_info["compare_config"]["required_aux_info"] = {
            GRAPH_STATUS: expected_status
        }
        expected = self._parse(self.golden)
        self.assertIsNone(expected.aux_info)
        self.comparer.compare_result(expected, self._parse(actual_payload))

    def _assert_compare_failed(self, expected_status, actual_payload, diff_field):
        with self.assertRaises(SmokeException) as raised:
            self._compare(expected_status, actual_payload)
        self.assertEqual(raised.exception.error_status, QueryStatus.COMPARE_FAILED)
        self.assertIn("required aux_info", raised.exception.message)
        self.assertIn(diff_field, raised.exception.message)

    def test_graph_status_survives_json_and_pydantic_parsing(self):
        for status in (REPLAYED, FALLBACK):
            with self.subTest(status=status):
                result = self._parse(self._response({GRAPH_STATUS: status}))
                self.assertEqual(result.model_dump()["aux_info"][GRAPH_STATUS], status)

    def test_matching_required_status_is_accepted_without_golden_aux_info(self):
        for status in (REPLAYED, FALLBACK):
            with self.subTest(status=status):
                self._compare(status, self._response({GRAPH_STATUS: status}))

    def test_mismatched_required_status_is_rejected_without_golden_aux_info(self):
        for expected, actual in (
            (REPLAYED, FALLBACK),
            (FALLBACK, REPLAYED),
            (REPLAYED, "not_requested"),
            (FALLBACK, "not_requested"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self._assert_compare_failed(
                    expected, self._response({GRAPH_STATUS: actual}), GRAPH_STATUS
                )

    def test_missing_required_status_is_rejected(self):
        for expected in (REPLAYED, FALLBACK):
            with self.subTest(expected=expected):
                self._assert_compare_failed(expected, self._response({}), GRAPH_STATUS)

    def test_missing_or_null_aux_info_is_rejected(self):
        for expected in (REPLAYED, FALLBACK):
            for actual in (self.golden, self._response(None)):
                with self.subTest(expected=expected, actual=actual):
                    self._assert_compare_failed(expected, actual, "missing")

    def test_legacy_golden_without_required_status_remains_compatible(self):
        for compare_config in ({}, {"required_aux_info": {}}):
            self.comparer.qr_info["compare_config"] = compare_config
            for status in (REPLAYED, FALLBACK):
                with self.subTest(compare_config=compare_config, actual_status=status):
                    self.comparer.compare_result(
                        self._parse(self.golden),
                        self._parse(self._response({GRAPH_STATUS: status})),
                    )

    def test_legacy_golden_without_compare_config_remains_compatible(self):
        del self.comparer.qr_info["compare_config"]
        self.comparer.compare_result(
            self._parse(self.golden),
            self._parse(self._response({GRAPH_STATUS: REPLAYED})),
        )


class OpenaiComparerAsymmetricCacheTest(unittest.TestCase):
    def setUp(self):
        fixture_dir = Path(__file__).resolve().parent.parent / "data/model/qwen25"
        self.fixtures = [
            json.loads(
                (
                    fixture_dir / f"q_r_l20_remote_cache_pd_asymmetric_{layout}.json"
                ).read_text()
            )
            for layout in ("tp", "cp")
        ]

    def _compare(self, row, content, aux_info):
        comparer = OpenaiComparer(None, "", row, Tracer(), False)
        expected = comparer.format_result(row["result"])
        actual = comparer.format_result(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 149,
                    "completion_tokens": 12,
                    "total_tokens": 161,
                },
                "aux_info": aux_info,
            }
        )
        comparer.compare_result(expected, actual)

    def test_valid_json_does_not_require_identical_generated_text(self):
        for fixture in self.fixtures:
            for row in fixture["query_result"]:
                for content in (
                    '{"recommendation":"Kung Fu Panda"}',
                    '{\n  "recommendation": "Zootopia"\n}',
                ):
                    with self.subTest(content=content, checks=row["compare_config"]):
                        aux = dict(row["compare_config"]["required_aux_info"])
                        aux.update(iter_count=12, output_len=12, cost_time=20.0)
                        self._compare(row, content, aux)

    def test_invalid_json_or_wrong_cache_attribution_is_rejected(self):
        for fixture in self.fixtures:
            row = fixture["query_result"][1]
            required = row["compare_config"]["required_aux_info"]
            invalid = [
                ("not JSON", required),
                ("{}", required),
                ('{"recommendation":"Zootopia"}', None),
            ]
            for field in (
                "prefill_remote_reuse_len",
                "decode_remote_reuse_len",
                "prefill_local_reuse_len",
                "decode_local_reuse_len",
            ):
                aux = dict(required)
                aux[field] = 0 if "remote" in field else 1
                invalid.append(('{"recommendation":"Zootopia"}', aux))
            for content, aux in invalid:
                with self.subTest(content=content, aux=aux):
                    with self.assertRaises(SmokeException) as raised:
                        self._compare(row, content, aux)
                    self.assertEqual(
                        raised.exception.error_status, QueryStatus.COMPARE_FAILED
                    )


if __name__ == "__main__":
    unittest.main()
