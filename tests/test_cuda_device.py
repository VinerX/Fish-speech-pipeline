import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

module_path = (
    Path(__file__).parents[1]
    / "fish_speech_lib/fish_speech/models/text2semantic/cuda_device.py"
)
spec = importlib.util.spec_from_file_location("cuda_device", module_path)
cuda_device = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda_device)
set_cuda_device_for_thread = cuda_device.set_cuda_device_for_thread


class CudaDeviceTests(unittest.TestCase):
    def test_both_queue_workers_set_cuda_device_before_model_load(self):
        source_path = (
            Path(__file__).parents[1]
            / "fish_speech_lib/fish_speech/models/text2semantic/inference.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        launches = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {
                "launch_thread_safe_queue",
                "launch_thread_safe_queue_agent",
            }
        }

        self.assertEqual(len(launches), 2)
        for launch in launches.values():
            worker = next(
                node
                for node in launch.body
                if isinstance(node, ast.FunctionDef) and node.name == "worker"
            )
            calls = {
                node.func.id: node.lineno
                for node in ast.walk(worker)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertLess(
                calls["set_cuda_device_for_thread"],
                calls["load_model"],
            )

    def test_selects_cuda_device_on_calling_thread(self):
        selected = SimpleNamespace(type="cuda", index=1)
        torch_module = SimpleNamespace(
            device=Mock(return_value=selected),
            cuda=SimpleNamespace(set_device=Mock()),
        )

        set_cuda_device_for_thread(torch_module, "cuda:1")

        torch_module.device.assert_called_once_with("cuda:1")
        torch_module.cuda.set_device.assert_called_once_with(selected)

    def test_leaves_cpu_device_unmodified(self):
        torch_module = SimpleNamespace(
            device=Mock(return_value=SimpleNamespace(type="cpu")),
            cuda=SimpleNamespace(set_device=Mock()),
        )

        set_cuda_device_for_thread(torch_module, "cpu")

        torch_module.cuda.set_device.assert_not_called()

    def test_leaves_unspecified_cuda_device_unmodified(self):
        torch_module = SimpleNamespace(
            device=Mock(return_value=SimpleNamespace(type="cuda", index=None)),
            cuda=SimpleNamespace(set_device=Mock()),
        )

        set_cuda_device_for_thread(torch_module, "cuda")

        torch_module.cuda.set_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
