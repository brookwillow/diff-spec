import json
import tempfile
import unittest
from pathlib import Path

from src.structured_drafter import (
    SLOT_NAMES,
    StructuredExample,
    build_label_space,
    labels_from_row,
    load_label_space,
    render_prediction,
)


class StructuredDrafterTests(unittest.TestCase):
    def setUp(self):
        self.schemas = {
            "ClimateControl": {
                "name": "ClimateControl",
                "inputSchema": {
                    "properties": {
                        "action": {"type": "string", "enum": ["打开", "关闭"]},
                        "device": {"type": "string", "enum": ["空调"]},
                        "feature": {"type": "string", "enum": ["温度"]},
                        "position": {"type": "string", "enum": ["主驾"]},
                    },
                    "required": ["action", "device"],
                },
            },
            "WindowControl": {
                "name": "WindowControl",
                "inputSchema": {
                    "properties": {
                        "action": {"type": "string", "enum": ["打开", "关闭"]},
                        "device": {"type": "string", "enum": ["车窗"]},
                    },
                    "required": ["action", "device"],
                },
            },
        }
        self.space = build_label_space(self.schemas)

    def test_build_label_space_includes_none_and_schema_values(self):
        self.assertEqual(self.space.kind_to_id["Action"], 0)
        self.assertIn("ClimateControl", self.space.tool_to_id)
        self.assertEqual(self.space.slot_value_to_id["action"]["NONE"], 0)
        self.assertIn("打开", self.space.slot_value_to_id["action"])
        self.assertIn("主驾", self.space.slot_value_to_id["position"])
        self.assertEqual(SLOT_NAMES, ("action", "device", "feature", "position", "value", "query", "index", "contact", "phone"))

    def test_labels_from_action_row_extracts_tool_and_slots(self):
        row = {
            "expected_type": "Action",
            "expected_tool_calls": [
                {
                    "name": "ClimateControl",
                    "arguments": {"action": "打开", "device": "空调", "feature": "温度"},
                }
            ],
        }

        labels = labels_from_row(row, self.space)

        self.assertEqual(labels.kind_id, self.space.kind_to_id["Action"])
        self.assertEqual(labels.tool_id, self.space.tool_to_id["ClimateControl"])
        self.assertEqual(labels.slot_ids["action"], self.space.slot_value_to_id["action"]["打开"])
        self.assertEqual(labels.slot_ids["device"], self.space.slot_value_to_id["device"]["空调"])
        self.assertEqual(labels.slot_ids["feature"], self.space.slot_value_to_id["feature"]["温度"])
        self.assertEqual(labels.slot_ids["position"], self.space.slot_value_to_id["position"]["NONE"])

    def test_render_prediction_builds_schema_valid_json_and_drops_unsupported_slots(self):
        example = StructuredExample(
            prompt="",
            kind_id=self.space.kind_to_id["Action"],
            tool_id=self.space.tool_to_id["ClimateControl"],
            slot_ids={
                "action": self.space.slot_value_to_id["action"]["打开"],
                "device": self.space.slot_value_to_id["device"]["空调"],
                "feature": self.space.slot_value_to_id["feature"]["温度"],
                "position": self.space.slot_value_to_id["position"]["主驾"],
                "value": self.space.slot_value_to_id["value"]["NONE"],
                "query": self.space.slot_value_to_id["query"]["NONE"],
                "index": self.space.slot_value_to_id["index"]["NONE"],
                "contact": self.space.slot_value_to_id["contact"]["NONE"],
                "phone": self.space.slot_value_to_id["phone"]["NONE"],
            },
        )

        output = render_prediction(example, self.space, self.schemas)

        self.assertEqual(
            json.loads(output),
            {"name": "ClimateControl", "arguments": {"action": "打开", "device": "空调", "feature": "温度", "position": "主驾"}},
        )

    def test_render_prediction_low_confidence_when_required_slot_missing(self):
        example = StructuredExample(
            prompt="",
            kind_id=self.space.kind_to_id["Action"],
            tool_id=self.space.tool_to_id["WindowControl"],
            slot_ids={slot: self.space.slot_value_to_id[slot]["NONE"] for slot in SLOT_NAMES},
        )

        output = render_prediction(example, self.space, self.schemas)

        self.assertEqual(output, "Reject")

    def test_load_label_space_round_trips_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "space.json"
            path.write_text(
                json.dumps(
                    {
                        "kind_to_id": self.space.kind_to_id,
                        "id_to_kind": self.space.id_to_kind,
                        "tool_to_id": self.space.tool_to_id,
                        "id_to_tool": self.space.id_to_tool,
                        "slot_value_to_id": self.space.slot_value_to_id,
                        "id_to_slot_value": self.space.id_to_slot_value,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            loaded = load_label_space(path)

        self.assertEqual(loaded.id_to_kind[0], "Action")
        self.assertEqual(loaded.id_to_tool[loaded.tool_to_id["ClimateControl"]], "ClimateControl")
        self.assertEqual(loaded.id_to_slot_value["action"][loaded.slot_value_to_id["action"]["打开"]], "打开")


if __name__ == "__main__":
    unittest.main()
