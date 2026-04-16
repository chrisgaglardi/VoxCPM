import unittest
from types import SimpleNamespace
from unittest import mock

from src.voxcpm.openai_server import Settings, VoxCPMOpenAIService


def _settings(**overrides):
    base = dict(
        model_source="openbmb/VoxCPM2",
        openai_model_name="voxcpm-tts",
        voice_library_dir="data/voices",
        host="0.0.0.0",
        port=3017,
        device="cpu",
        load_denoiser=False,
        optimize=False,
        preload_model=False,
        idle_unload_seconds=300,
        idle_check_interval_seconds=15,
        default_cfg_value=2.0,
        default_inference_timesteps=10,
        default_max_len=4096,
    )
    base.update(overrides)
    return Settings(**base)


class IdleUnloadTests(unittest.TestCase):
    def test_unload_model_if_idle_drops_loaded_model(self):
        service = VoxCPMOpenAIService(_settings(idle_unload_seconds=300), SimpleNamespace())
        service._model = object()
        service._last_used_monotonic = 10.0

        with (
            mock.patch("src.voxcpm.openai_server.torch.cuda.is_available", return_value=True),
            mock.patch("src.voxcpm.openai_server.torch.cuda.empty_cache") as empty_cache,
            mock.patch("src.voxcpm.openai_server.torch.cuda.ipc_collect") as ipc_collect,
        ):
            unloaded = service.unload_model_if_idle(now=311.0)

        self.assertTrue(unloaded)
        self.assertIsNone(service._model)
        empty_cache.assert_called_once_with()
        ipc_collect.assert_called_once_with()

    def test_unload_model_if_idle_keeps_recent_model(self):
        service = VoxCPMOpenAIService(_settings(idle_unload_seconds=300), SimpleNamespace())
        marker = object()
        service._model = marker
        service._last_used_monotonic = 10.0

        with mock.patch("src.voxcpm.openai_server.torch.cuda.is_available", return_value=False):
            unloaded = service.unload_model_if_idle(now=200.0)

        self.assertFalse(unloaded)
        self.assertIs(service._model, marker)


if __name__ == "__main__":
    unittest.main()
