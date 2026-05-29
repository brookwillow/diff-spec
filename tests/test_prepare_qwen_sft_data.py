import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_qwen_sft_data import collect_sft_rows, is_trainable_messages, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


class PrepareQwenSftDataTests(unittest.TestCase):
    def test_keeps_direct_user_to_assistant_tool_call(self):
        messages = [
            {"role": "user", "content": "打开空调"},
            {
                "role": "assistant",
                "content": '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}',
            },
        ]

        self.assertTrue(is_trainable_messages(messages))

    def test_keeps_multiturn_without_tool_role(self):
        messages = [
            {"role": "user", "content": "车里有点热"},
            {"role": "assistant", "content": "帮您调低空调温度。"},
            {"role": "user", "content": "再低一点"},
            {
                "role": "assistant",
                "content": '{"name":"ClimateControl","arguments":{"action":"调低","device":"空调","feature":"温度"}}',
            },
        ]

        self.assertTrue(is_trainable_messages(messages))

    def test_filters_tool_execution_transcripts(self):
        messages = [
            {
                "role": "assistant",
                "content": '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}',
            },
            {"role": "tool", "content": '{"status":"success"}'},
            {"role": "assistant", "content": "好的，已为您打开空调。"},
        ]

        self.assertFalse(is_trainable_messages(messages))

    def test_collect_sft_rows_from_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            output = root / "train.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": "打开空调"},
                                    {
                                        "role": "assistant",
                                        "content": '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}',
                                    },
                                ]
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "assistant", "content": '{"name":"ClimateControl","arguments":{}}'},
                                    {"role": "tool", "content": '{"status":"success"}'},
                                    {"role": "assistant", "content": "好的"},
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = collect_sft_rows([source])
            write_jsonl(rows, output)
            written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(written[0]["messages"][0]["content"], "打开空调")

    def test_qwen_sft_config_uses_full_system_prompt(self):
        config = (REPO_ROOT / "configs" / "qwen_sft_lora.yaml").read_text(encoding="utf-8")

        self.assertIn("system: data/system-prompt.txt", config)
        self.assertIn("max_length: 4096", config)
        self.assertIn("loss_scale: last_round", config)
        prompt = (REPO_ROOT / "data" / "system-prompt.txt").read_text(encoding="utf-8")
        self.assertGreater(len(prompt), 5000)


if __name__ == "__main__":
    unittest.main()
