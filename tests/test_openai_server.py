import unittest
from types import SimpleNamespace
from unittest import mock

from src.voxcpm.openai_server import (
    Settings,
    VoxCPMOpenAIService,
    _infer_upload_suffix,
    _normalize_audio_for_voxcpm,
)


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


class UploadHandlingTests(unittest.TestCase):
    def test_infer_upload_suffix_uses_content_type_when_filename_has_no_extension(self):
        upload = SimpleNamespace(filename="recording", content_type="audio/mp4")

        self.assertEqual(_infer_upload_suffix(upload), ".m4a")

    def test_normalize_audio_returns_original_when_ffmpeg_is_missing(self):
        with mock.patch("src.voxcpm.openai_server.shutil.which", return_value=None):
            self.assertEqual(_normalize_audio_for_voxcpm("sample.m4a"), "sample.m4a")

    def test_normalize_audio_raises_clear_error_when_ffmpeg_decode_fails(self):
        created_temp = SimpleNamespace(name="normalized.wav")

        with (
            mock.patch("src.voxcpm.openai_server.shutil.which", return_value="/usr/bin/ffmpeg"),
            mock.patch("src.voxcpm.openai_server.tempfile.NamedTemporaryFile") as named_temp,
            mock.patch("src.voxcpm.openai_server.subprocess.run") as run,
            mock.patch("src.voxcpm.openai_server.os.remove") as remove,
        ):
            named_temp.return_value.__enter__.return_value = created_temp
            run.return_value = SimpleNamespace(returncode=1, stderr=b"bad audio stream")

            with self.assertRaisesRegex(ValueError, "Uploaded audio could not be decoded"):
                _normalize_audio_for_voxcpm("sample.m4a")

        remove.assert_called_once_with("normalized.wav")

    def test_normalize_audio_returns_wav_path_on_success(self):
        created_temp = SimpleNamespace(name="normalized.wav")

        with (
            mock.patch("src.voxcpm.openai_server.shutil.which", return_value="/usr/bin/ffmpeg"),
            mock.patch("src.voxcpm.openai_server.tempfile.NamedTemporaryFile") as named_temp,
            mock.patch("src.voxcpm.openai_server.subprocess.run") as run,
        ):
            named_temp.return_value.__enter__.return_value = created_temp
            run.return_value = SimpleNamespace(returncode=0, stderr=b"")

            normalized = _normalize_audio_for_voxcpm("sample.m4a")

        self.assertEqual(normalized, "normalized.wav")


if __name__ == "__main__":
    unittest.main()
