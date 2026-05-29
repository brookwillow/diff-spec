import unittest

from src.diffusion_drafter import MaskedDrafterConfig, build_masked_example, trim_decoded_prediction
from scripts.train_diffusion_drafter import parse_simple_config


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    mask_token_id = 103

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class DiffusionDrafterTests(unittest.TestCase):
    def test_build_masked_example_supervises_only_target_span(self):
        tokenizer = FakeTokenizer()
        config = MaskedDrafterConfig(max_length=17, max_target_tokens=4)

        example = build_masked_example(tokenizer, "用户:打开空调\n", "OK", config)

        self.assertEqual(example["input_ids"][-8:], [102, 103, 103, 103, 102, 0, 0, 0])
        self.assertEqual(example["labels"][-8:], [-100, ord("O"), ord("K"), 102, -100, -100, -100, -100])
        self.assertEqual(example["attention_mask"][-3:], [0, 0, 0])

    def test_build_masked_example_truncates_prompt_from_left(self):
        tokenizer = FakeTokenizer()
        config = MaskedDrafterConfig(max_length=9, max_target_tokens=2)

        example = build_masked_example(tokenizer, "abcdef", "xy", config)

        self.assertEqual(example["input_ids"], [101, ord("d"), ord("e"), ord("f"), 102, 103, 103, 103, 102])
        self.assertEqual(example["labels"], [-100, -100, -100, -100, -100, ord("x"), ord("y"), 102, -100])

    def test_trim_decoded_prediction_stops_at_special_tokens_and_role_marker(self):
        text = ' {"name":"A"} [SEP] Assistant: ignored'

        self.assertEqual(trim_decoded_prediction(text), '{"name":"A"}')

    def test_parse_simple_config_without_yaml_dependency(self):
        config = parse_simple_config("model: hfl/chinese-macbert-base\nmax_length: 512\nlearning_rate: 0.00005\n")

        self.assertEqual(config["model"], "hfl/chinese-macbert-base")
        self.assertEqual(config["max_length"], 512)
        self.assertEqual(config["learning_rate"], 0.00005)


if __name__ == "__main__":
    unittest.main()
