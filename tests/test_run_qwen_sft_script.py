import os
import subprocess
import io
import unittest
from unittest import mock

from scripts.run_qwen_sft import detect_torch_device


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_qwen_sft.sh")
PY_SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_qwen_sft.py")


def run_script_for_device(device: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "SKIP_PREPARE": "1",
            "SWIFT_DEVICE": device,
        }
    )
    return subprocess.run(
        ["python3", PY_SCRIPT_PATH],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class RunQwenSftScriptTest(unittest.TestCase):
    def test_python_runner_exists(self):
        self.assertTrue(os.path.exists(PY_SCRIPT_PATH))

    def test_python_runner_has_valid_syntax(self):
        result = subprocess.run(
            ["python3", "-m", "py_compile", PY_SCRIPT_PATH],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_script_has_valid_bash_syntax(self):
        result = subprocess.run(
            ["bash", "-n", SCRIPT_PATH],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mps_device_uses_macos_gpu_safe_overrides(self):
        result = run_script_for_device("mps")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Detected device: mps", result.stdout)
        self.assertIn("PYTORCH_ENABLE_MPS_FALLBACK=1", result.stdout)
        self.assertIn("--device_map mps:0", result.stdout)
        self.assertIn("--torch_dtype float32", result.stdout)
        self.assertIn("--fp16 false", result.stdout)
        self.assertIn("--bf16 false", result.stdout)
        self.assertIn("--attn_impl eager", result.stdout)

    def test_cpu_device_uses_cpu_safe_overrides(self):
        result = run_script_for_device("cpu")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Detected device: cpu", result.stdout)
        self.assertIn("--device_map cpu", result.stdout)
        self.assertIn("--torch_dtype float32", result.stdout)
        self.assertIn("--fp16 false", result.stdout)
        self.assertIn("--bf16 false", result.stdout)

    def test_cuda_device_keeps_swift_default_cuda_path(self):
        result = run_script_for_device("cuda")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Detected device: cuda", result.stdout)
        self.assertIn("swift sft configs/qwen_sft_lora.yaml", result.stdout)
        self.assertNotIn("--device_map", result.stdout)
        self.assertNotIn("--torch_dtype float32", result.stdout)

    def test_device_auto_detection_produces_a_supported_device(self):
        env = os.environ.copy()
        env.update({"DRY_RUN": "1", "SKIP_PREPARE": "1"})
        env.pop("SWIFT_DEVICE", None)

        result = subprocess.run(
            ["python3", PY_SCRIPT_PATH],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"Detected device: (cuda|mps|cpu)")

    def test_missing_torch_reports_install_hint(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'torch'\n",
        )

        captured = io.StringIO()
        with mock.patch("subprocess.run", return_value=completed), mock.patch("sys.stderr", captured):
            with self.assertRaises(SystemExit) as context:
                detect_torch_device("qwen-omni")

        self.assertEqual(context.exception.code, 1)
        self.assertIn("PyTorch is not installed", captured.getvalue())
        self.assertIn("conda run -n qwen-omni pip install torch torchvision torchaudio", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
