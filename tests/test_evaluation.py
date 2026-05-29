import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import (
    Evaluator,
    OutputKind,
    canonicalize_output,
    load_jsonl,
    load_tool_schemas,
    parse_assistant_output,
)


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.schemas = load_tool_schemas(ROOT / "data" / "tools.json")
        self.evaluator = Evaluator(self.schemas)

    def test_parse_single_tool_call(self):
        parsed = parse_assistant_output(
            '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}'
        )

        self.assertEqual(parsed.kind, OutputKind.TOOL_CALLS)
        self.assertEqual(parsed.tool_calls[0]["name"], "ClimateControl")

    def test_parse_multi_tool_call_array(self):
        parsed = parse_assistant_output(
            '[{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}},'
            '{"name":"LightControl","arguments":{"action":"关闭","device":"阅读灯"}}]'
        )

        self.assertEqual(parsed.kind, OutputKind.TOOL_CALLS)
        self.assertEqual(len(parsed.tool_calls), 2)

    def test_parse_reject_and_text(self):
        self.assertEqual(parse_assistant_output("Reject").kind, OutputKind.REJECT)
        self.assertEqual(parse_assistant_output("请问您要调高还是调低？").kind, OutputKind.TEXT)

    def test_validates_schema_required_and_unknown_fields(self):
        valid = {"name": "ClimateControl", "arguments": {"action": "打开", "device": "空调"}}
        invalid = {"name": "ClimateControl", "arguments": {"device": "空调", "extra": "x"}}

        self.assertTrue(self.evaluator.validate_tool_call(valid).valid)
        result = self.evaluator.validate_tool_call(invalid)
        self.assertFalse(result.valid)
        self.assertIn("missing required argument: action", result.errors)
        self.assertIn("unknown argument: extra", result.errors)

    def test_numeric_value_is_allowed_for_vehicle_adjustment(self):
        call = {
            "name": "ClimateControl",
            "arguments": {
                "action": "调到",
                "device": "空调",
                "feature": "温度",
                "value": "20.5",
            },
        }

        self.assertTrue(self.evaluator.validate_tool_call(call).valid)

    def test_exact_match_uses_canonical_json(self):
        expected = '{"name":"ClimateControl","arguments":{"device":"空调","action":"打开"}}'
        predicted = '{ "arguments": { "action": "打开", "device": "空调" }, "name": "ClimateControl" }'

        self.assertEqual(canonicalize_output(expected), canonicalize_output(predicted))
        self.assertTrue(self.evaluator.score_pair(expected, predicted).exact_match)

    def test_evaluate_prediction_file(self):
        gold = ROOT / "data" / "splits" / "by_tool" / "ClimateControl.jsonl"
        first = load_jsonl(gold)[0]
        expected = first["messages"][-1]["content"]

        with tempfile.TemporaryDirectory() as tmp:
            pred_path = Path(tmp) / "predictions.jsonl"
            pred_path.write_text(
                json.dumps({"prediction": expected}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            summary = self.evaluator.evaluate_files(gold, pred_path, limit=1)

        self.assertEqual(summary.total, 1)
        self.assertEqual(summary.exact_match_rate, 1.0)
        self.assertEqual(summary.schema_valid_rate, 1.0)

    def test_detailed_summary_tracks_class_tool_param_and_json_errors(self):
        gold_rows = [
            {
                "expected_type": "Action",
                "expected_tool_calls": [
                    {"name": "ClimateControl", "arguments": {"action": "打开", "device": "空调"}}
                ],
            },
            {
                "expected_type": "Action",
                "expected_tool_calls": [
                    {"name": "WindowControl", "arguments": {"action": "打开", "device": "车窗"}}
                ],
            },
            {"expected_type": "Reject", "expected_tool_calls": []},
            {"expected_type": "Clarify", "expected_tool_calls": []},
        ]
        predictions = [
            {"prediction": '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}'},
            {"prediction": '{"name":"WindowControl","arguments":{"action":"关闭","device":"车窗"}}'},
            {"prediction": '{"name":"Reject"'},
            {"prediction": "您要调哪个位置？"},
        ]

        summary = self.evaluator.evaluate_rows_detailed(gold_rows, predictions).as_dict()

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["classification_accuracy"], 0.75)
        self.assertEqual(summary["json_format_error_rate"], 0.25)
        self.assertEqual(summary["action_rows"], 2)
        self.assertEqual(summary["tool_selection_accuracy"], 1.0)
        self.assertEqual(summary["parameter_fill_accuracy"], 0.5)

    def test_validate_dataset_accepts_gold_text_reject_and_tools(self):
        paths = [
            ROOT / "data" / "splits" / "by_tool" / "ClimateControl.jsonl",
            ROOT / "data" / "splits" / "clarify.jsonl",
            ROOT / "data" / "splits" / "reject.jsonl",
        ]

        summary = self.evaluator.validate_dataset(paths)

        self.assertGreater(summary.total, 0)
        self.assertEqual(summary.invalid_json, 0)
        self.assertEqual(summary.schema_valid_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
