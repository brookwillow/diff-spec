import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_structured_drafter_data import collect_structured_rows


class PrepareStructuredDrafterDataTests(unittest.TestCase):
    def test_collect_structured_rows_renders_prompt_and_labels(self):
        schemas = {
            "AppControl": {
                "name": "AppControl",
                "inputSchema": {
                    "properties": {
                        "action": {"type": "string", "enum": ["打开"]},
                        "feature": {"type": "string", "enum": ["导航地图"]},
                    },
                    "required": ["action", "feature"],
                },
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "app_001",
                        "messages": [
                            {"role": "user", "content": "打开导航地图"},
                            {
                                "role": "assistant",
                                "content": '{"name":"AppControl","arguments":{"action":"打开","feature":"导航地图"}}',
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            rows, space = collect_structured_rows([source], schemas, "SYSTEM")

        self.assertEqual(len(rows), 1)
        self.assertIn("User:\n打开导航地图", rows[0]["prompt"])
        self.assertNotIn("System:", rows[0]["prompt"])
        self.assertEqual(rows[0]["kind_id"], space.kind_to_id["Action"])
        self.assertEqual(rows[0]["tool_id"], space.tool_to_id["AppControl"])
        self.assertEqual(rows[0]["slot_ids"]["action"], space.slot_value_to_id["action"]["打开"])
        self.assertEqual(rows[0]["slot_ids"]["feature"], space.slot_value_to_id["feature"]["导航地图"])


if __name__ == "__main__":
    unittest.main()
