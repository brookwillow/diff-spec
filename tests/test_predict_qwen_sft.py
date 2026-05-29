import unittest
import sys
import types
from collections import UserDict
from unittest import mock

from scripts.predict_qwen_sft import build_generation_inputs, prompt_messages


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeTorch(types.SimpleNamespace):
    @staticmethod
    def ones_like(tensor):
        return FakeTensor([[1] * tensor.shape[1] for _ in range(tensor.shape[0])])


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

    def test_build_generation_inputs_accepts_batch_encoding_like_output(self):
        tokenizer = mock.Mock()
        fake = FakeTensor([[1, 2, 3]])
        tokenizer.apply_chat_template.return_value = UserDict({"input_ids": fake})
        model = mock.Mock()
        model.device = "cpu"

        with mock.patch.dict(sys.modules, {"torch": FakeTorch()}):
            inputs = build_generation_inputs(model, tokenizer, [{"role": "user", "content": "打开空调"}])

        self.assertIs(inputs["input_ids"], fake)
        self.assertEqual(inputs["attention_mask"].values, [[1, 1, 1]])

    def test_build_generation_inputs_accepts_tensor_output(self):
        tokenizer = mock.Mock()
        fake = FakeTensor([[4, 5]])
        tokenizer.apply_chat_template.return_value = fake
        model = mock.Mock()
        model.device = "cpu"

        with mock.patch.dict(sys.modules, {"torch": FakeTorch()}):
            inputs = build_generation_inputs(model, tokenizer, [{"role": "user", "content": "打开空调"}])

        self.assertIs(inputs["input_ids"], fake)
        self.assertEqual(inputs["attention_mask"].values, [[1, 1]])


if __name__ == "__main__":
    unittest.main()
