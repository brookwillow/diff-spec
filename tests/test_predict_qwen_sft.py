import unittest

from scripts.predict_qwen_sft import prompt_messages


class PredictQwenSftTests(unittest.TestCase):
    def test_prompt_messages_remove_gold_assistant_and_inject_system(self):
        row = {
            "messages": [
                {"role": "user", "content": "打开空调"},
                {"role": "assistant", "content": '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}'},
            ]
        }

        messages = prompt_messages(row, "SYSTEM")

        self.assertEqual(messages, [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "打开空调"}])

    def test_prompt_messages_replace_existing_system(self):
        row = {
            "messages": [
                {"role": "system", "content": "OLD"},
                {"role": "user", "content": "打开空调"},
            ]
        }

        messages = prompt_messages(row, "NEW")

        self.assertEqual(messages[0], {"role": "system", "content": "NEW"})


if __name__ == "__main__":
    unittest.main()
