import unittest

from scripts.serve_comparison import _format_output


class ServeComparisonTimingTests(unittest.TestCase):
    def test_speculative_output_shows_continuation_generation_time(self):
        rendered = _format_output(
            {
                "prediction": '{"name":"ClimateControl","arguments":{"device":"空调","action":"打开"}}',
                "latency_ms": 626.8,
                "draft_text": '{"name":"bad","arguments":{}}',
                "draft_ms": 10.9,
                "prep_ms": 1.2,
                "verify_ms": 89.8,
                "accept_check_ms": 0.5,
                "continuation_ms": 524.1,
                "decode_ms": 0.2,
                "other_ms": 0.1,
                "accepted_tokens": 3,
                "draft_tokens": 16,
                "accept_status": "partial",
            },
            "Diffusion Spec",
        )

        self.assertIn("626.8 ms", rendered)
        self.assertIn("Verify forward: 89.8ms", rendered)
        self.assertIn("Continue gen: 524.1ms", rendered)
        self.assertIn("Other: 0.1ms", rendered)


if __name__ == "__main__":
    unittest.main()
