import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_diffusion_data import collect_diffusion_rows, render_prompt


class PrepareDiffusionDataTests(unittest.TestCase):
    def test_render_prompt_keeps_history_and_drops_final_assistant(self):
        messages = [
            {"role": "system", "content": "OLD"},
            {"role": "user", "content": "车里有点热"},
            {"role": "assistant", "content": "帮您调低空调温度。"},
            {"role": "user", "content": "再低一点"},
            {
                "role": "assistant",
                "content": '{"name":"ClimateControl","arguments":{"action":"调低","device":"空调","feature":"温度"}}',
            },
        ]

        prompt = render_prompt(messages, "SYSTEM")

        self.assertIn("System:\nSYSTEM", prompt)
        self.assertIn("User:\n车里有点热", prompt)
        self.assertIn("Assistant:\n帮您调低空调温度。", prompt)
        self.assertIn("User:\n再低一点", prompt)
        self.assertTrue(prompt.endswith("Assistant:\n"))
        self.assertNotIn('"name":"ClimateControl"', prompt)

    def test_collect_diffusion_rows_extracts_final_assistant_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "id": "x1",
                        "expected_type": "Action",
                        "expected_tool_calls": [{"name": "ClimateControl", "arguments": {"action": "打开", "device": "空调"}}],
                        "messages": [
                            {"role": "user", "content": "打开空调"},
                            {
                                "role": "assistant",
                                "content": '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}',
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            rows = collect_diffusion_rows([source], system_prompt="SYSTEM")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "x1")
        self.assertEqual(rows[0]["kind"], "Action")
        self.assertEqual(rows[0]["tool_name"], "ClimateControl")
        self.assertEqual(rows[0]["target"], '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}')
        self.assertIn("User:\n打开空调", rows[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
