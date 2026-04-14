import argparse
import asyncio
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .core import VoxCPM


logger = logging.getLogger("voxcpm.openai")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class Settings:
    model_source: str
    openai_model_name: str
    voice_library_dir: str
    host: str
    port: int
    device: str | None
    load_denoiser: bool
    optimize: bool
    preload_model: bool
    default_cfg_value: float
    default_inference_timesteps: int
    default_max_len: int

    @classmethod
    def from_env(cls) -> "Settings":
        model_source = os.getenv("VOXCPM_MODEL", "openbmb/VoxCPM2")
        return cls(
            model_source=model_source,
            openai_model_name=os.getenv("VOXCPM_OPENAI_MODEL_NAME", "voxcpm-tts"),
            voice_library_dir=os.getenv("VOXCPM_VOICE_LIBRARY_DIR", "data/voices"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_env_int("PORT", 3017),
            device=os.getenv("VOXCPM_DEVICE", "cuda"),
            load_denoiser=_env_flag("VOXCPM_LOAD_DENOISER", False),
            optimize=_env_flag("VOXCPM_OPTIMIZE", False),
            preload_model=_env_flag("VOXCPM_PRELOAD_MODEL", True),
            default_cfg_value=_env_float("VOXCPM_CFG_VALUE", 2.0),
            default_inference_timesteps=_env_int("VOXCPM_INFERENCE_TIMESTEPS", 10),
            default_max_len=_env_int("VOXCPM_MAX_LEN", 4096),
        )


SETTINGS = Settings.from_env()
STATIC_DIR = Path(__file__).with_name("static")


@dataclass(frozen=True)
class VoiceRecord:
    id: str
    name: str
    filename: str
    created_at: str
    updated_at: str

    def to_public_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class VoiceRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class VoiceLibrary:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = Path(root_dir)
        self.files_dir = self.root_dir / "files"
        self.index_path = self.root_dir / "voices.json"
        self._lock = threading.Lock()
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def list_voices(self) -> list[dict[str, str]]:
        with self._lock:
            voices = [self._record_from_entry(entry).to_public_dict() for entry in self._load_entries()]
        return sorted(voices, key=lambda voice: voice["name"].lower())

    def add_voice(self, name: str, source_path: str, original_filename: str | None = None) -> dict[str, str]:
        cleaned_name = self._clean_name(name)
        source = Path(source_path)
        suffix = Path(original_filename or source.name).suffix.lower() or ".wav"
        now = self._timestamp()
        voice_id = uuid4().hex
        filename = f"{voice_id}{suffix}"
        destination = self.files_dir / filename

        with self._lock:
            entries = self._load_entries()
            self._ensure_unique_name(entries, cleaned_name)
            shutil.move(str(source), destination)
            entry = {
                "id": voice_id,
                "name": cleaned_name,
                "filename": filename,
                "created_at": now,
                "updated_at": now,
            }
            entries.append(entry)
            self._save_entries(entries)

        return self._record_from_entry(entry).to_public_dict()

    def rename_voice(self, voice_id: str, new_name: str) -> dict[str, str]:
        cleaned_name = self._clean_name(new_name)
        with self._lock:
            entries = self._load_entries()
            entry = self._find_by_id(entries, voice_id)
            if entry is None:
                raise FileNotFoundError(f"Saved voice '{voice_id}' was not found")
            self._ensure_unique_name(entries, cleaned_name, ignored_id=voice_id)
            entry["name"] = cleaned_name
            entry["updated_at"] = self._timestamp()
            self._save_entries(entries)
            return self._record_from_entry(entry).to_public_dict()

    def delete_voice(self, voice_id: str) -> None:
        file_path: Path | None = None
        with self._lock:
            entries = self._load_entries()
            entry = self._find_by_id(entries, voice_id)
            if entry is None:
                raise FileNotFoundError(f"Saved voice '{voice_id}' was not found")
            file_path = self.files_dir / entry["filename"]
            remaining = [item for item in entries if item["id"] != voice_id]
            self._save_entries(remaining)

        if file_path is not None:
            file_path.unlink(missing_ok=True)

    def resolve_name(self, name: str) -> str:
        normalized_name = self._normalize_name(name)
        with self._lock:
            entries = self._load_entries()
            for entry in entries:
                if self._normalize_name(entry["name"]) == normalized_name:
                    file_path = self.files_dir / entry["filename"]
                    if not file_path.exists():
                        raise FileNotFoundError(
                            f"Saved voice '{entry['name']}' is registered but the audio file is missing"
                        )
                    return str(file_path)
        raise FileNotFoundError(f"Saved voice '{name}' was not found")

    def resolve_exact_name(self, name: str) -> str | None:
        candidate = name.strip()
        if not candidate:
            return None
        with self._lock:
            entries = self._load_entries()
            for entry in entries:
                if entry["name"] != candidate:
                    continue
                file_path = self.files_dir / entry["filename"]
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Saved voice '{entry['name']}' is registered but the audio file is missing"
                    )
                return str(file_path)
        return None

    def _load_entries(self) -> list[dict[str, str]]:
        if not self.index_path.exists():
            return []
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Voice library metadata is unreadable: {exc}") from exc
        voices = payload.get("voices", [])
        if not isinstance(voices, list):
            raise ValueError("Voice library metadata is invalid")
        return voices

    def _save_entries(self, entries: list[dict[str, str]]) -> None:
        payload = {"voices": entries}
        temp_path = self.index_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.index_path)

    def _record_from_entry(self, entry: dict[str, str]) -> VoiceRecord:
        return VoiceRecord(
            id=entry["id"],
            name=entry["name"],
            filename=entry["filename"],
            created_at=entry["created_at"],
            updated_at=entry["updated_at"],
        )

    def _find_by_id(self, entries: list[dict[str, str]], voice_id: str) -> dict[str, str] | None:
        for entry in entries:
            if entry["id"] == voice_id:
                return entry
        return None

    def _ensure_unique_name(
        self,
        entries: list[dict[str, str]],
        candidate_name: str,
        ignored_id: str | None = None,
    ) -> None:
        normalized_candidate = self._normalize_name(candidate_name)
        for entry in entries:
            if ignored_id and entry["id"] == ignored_id:
                continue
            if self._normalize_name(entry["name"]) == normalized_candidate:
                raise ValueError(f"A saved voice named '{candidate_name}' already exists")

    def _clean_name(self, raw_name: str) -> str:
        cleaned_name = raw_name.strip()
        if not cleaned_name:
            raise ValueError("Saved voice name is required")
        if len(cleaned_name) > 80:
            raise ValueError("Saved voice name must be 80 characters or fewer")
        forbidden = {"/", "\\", "\n", "\r", "\t"}
        if any(char in cleaned_name for char in forbidden):
            raise ValueError("Saved voice name cannot contain slashes or control characters")
        return cleaned_name

    def _normalize_name(self, value: str) -> str:
        return value.strip().casefold()

    def _timestamp(self) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class SpeechRequest(BaseModel):
    model: str = Field(..., description="OpenAI-compatible model id")
    input: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str | None = Field(
        default=None,
        description="Used as a VoxCPM voice description for OpenAI compatibility",
    )
    instructions: str | None = Field(
        default=None,
        description="Preferred field for voice design instructions",
    )
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    cfg_value: float | None = Field(default=None, ge=0.1, le=10.0)
    inference_timesteps: int | None = Field(default=None, ge=1, le=100)
    max_len: int | None = Field(default=None, ge=32, le=16384)
    normalize: bool = False
    denoise: bool = False
    control: str | None = Field(
        default=None,
        description="Explicit VoxCPM voice description. Overrides voice/instructions.",
    )
    saved_voice: str | None = Field(
        default=None,
        description="Name of a stored voice library entry to use as reference audio.",
    )
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    reference_audio_path: str | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "SpeechRequest":
        if self.prompt_audio_path and not self.prompt_text:
            raise ValueError("prompt_text is required when prompt_audio_path is provided")
        if self.prompt_text and not self.prompt_audio_path:
            raise ValueError("prompt_audio_path is required when prompt_text is provided")
        if self.saved_voice and self.reference_audio_path:
            raise ValueError("saved_voice cannot be combined with a direct reference audio upload")
        if self.response_format.lower() not in {"wav", "flac", "pcm", "mp3"}:
            raise ValueError("response_format must be one of: wav, flac, pcm, mp3")
        return self


class VoxCPMOpenAIService:
    def __init__(self, settings: Settings, voice_library: VoiceLibrary) -> None:
        self.settings = settings
        self.voice_library = voice_library
        self._model: VoxCPM | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    @property
    def accepted_model_names(self) -> set[str]:
        names = {self.settings.openai_model_name, self.settings.model_source}
        local_name = os.path.basename(self.settings.model_source.rstrip("/\\"))
        if local_name:
            names.add(local_name)
        return names

    def ensure_model_loaded(self) -> VoxCPM:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            logger.info(
                "Loading VoxCPM model from %s on device=%s",
                self.settings.model_source,
                self.settings.device,
            )
            self._model = VoxCPM.from_pretrained(
                self.settings.model_source,
                load_denoiser=self.settings.load_denoiser,
                optimize=self.settings.optimize,
                device=self.settings.device,
            )
            logger.info("Model loaded")
            return self._model

    def describe_models(self) -> list[dict[str, str]]:
        return [
            {
                "id": self.settings.openai_model_name,
                "object": "model",
                "owned_by": "openbmb",
            }
        ]

    def _resolve_reference_audio(self, request: SpeechRequest) -> tuple[str | None, bool]:
        if request.saved_voice:
            return self.voice_library.resolve_name(request.saved_voice), False
        if request.reference_audio_path:
            return request.reference_audio_path, False
        if request.voice:
            matched_path = self.voice_library.resolve_exact_name(request.voice)
            if matched_path is not None:
                return matched_path, True
        return None, False

    def _final_text(self, request: SpeechRequest, consumed_voice_as_saved_voice: bool = False) -> str:
        voice_control = None if consumed_voice_as_saved_voice else request.voice
        control = request.control or request.instructions or voice_control
        if request.prompt_text and control:
            raise ValueError("control/instructions/voice cannot be combined with prompt cloning")
        if control:
            return f"({control.strip()}){request.input.strip()}"
        return request.input.strip()

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        if request.model not in self.accepted_model_names:
            raise ValueError(
                f"Unknown model '{request.model}'. Available model: {self.settings.openai_model_name}"
            )

        model = self.ensure_model_loaded()
        reference_audio_path, consumed_voice_as_saved_voice = self._resolve_reference_audio(request)
        final_text = self._final_text(request, consumed_voice_as_saved_voice=consumed_voice_as_saved_voice)
        cfg_value = request.cfg_value or self.settings.default_cfg_value
        inference_timesteps = request.inference_timesteps or self.settings.default_inference_timesteps
        max_len = request.max_len or self.settings.default_max_len

        with self._generate_lock:
            wav = model.generate(
                text=final_text,
                prompt_wav_path=request.prompt_audio_path,
                prompt_text=request.prompt_text,
                reference_wav_path=reference_audio_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_len=max_len,
                normalize=request.normalize,
                denoise=request.denoise,
            )

        wav = np.asarray(wav, dtype=np.float32)
        if request.speed != 1.0:
            wav = librosa.effects.time_stretch(wav, rate=request.speed).astype(np.float32, copy=False)

        return _encode_audio(wav, model.tts_model.sample_rate, request.response_format.lower())


def _encode_audio(wav: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    if response_format == "pcm":
        pcm16 = np.clip(wav, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2")
        return pcm16.tobytes(), "application/octet-stream"

    buffer = io.BytesIO()
    if response_format == "wav":
        sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), "audio/wav"
    if response_format == "flac":
        sf.write(buffer, wav, sample_rate, format="FLAC")
        return buffer.getvalue(), "audio/flac"
    if response_format == "mp3":
        return _encode_mp3(wav, sample_rate), "audio/mpeg"

    raise ValueError(f"Unsupported response_format: {response_format}")


def _encode_mp3(wav: np.ndarray, sample_rate: int) -> bytes:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise ValueError("response_format='mp3' requires ffmpeg to be installed and available on PATH")

    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, wav, sample_rate, format="WAV", subtype="PCM_16")
    result = subprocess.run(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "pipe:1",
        ],
        input=wav_bytes.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        error_msg = result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg mp3 encode failed"
        raise ValueError(error_msg)
    return result.stdout


def _openai_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": None,
                "code": None,
            }
        },
    )


voice_library = VoiceLibrary(SETTINGS.voice_library_dir)
service = VoxCPMOpenAIService(SETTINGS, voice_library)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    if SETTINGS.preload_model:
        await asyncio.to_thread(service.ensure_model_loaded)
    yield


app = FastAPI(
    title="VoxCPM OpenAI-Compatible API",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/ui/static", StaticFiles(directory=STATIC_DIR), name="ui-static")


def _parse_bool_form(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _save_upload(upload: UploadFile | None) -> str | None:
    if upload is None or not upload.filename:
        return None

    suffix = Path(upload.filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            temp_file.write(chunk)
    await upload.close()
    return temp_path


def _cleanup_temp_files(paths: list[str | None]) -> None:
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
async def ui_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
async def healthz() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": service._model is not None}


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    return {"object": "list", "data": service.describe_models()}


@app.get("/v1/voices")
async def list_voices() -> dict[str, object]:
    try:
        voices = await asyncio.to_thread(voice_library.list_voices)
    except ValueError as exc:
        return _openai_error(str(exc), status_code=500)
    return {"object": "list", "data": voices}


@app.post("/v1/voices", response_model=None)
async def create_voice(name: str = Form(...), audio: UploadFile = File(...)) -> JSONResponse:
    temp_path: str | None = None
    try:
        temp_path = await _save_upload(audio)
        if temp_path is None:
            raise ValueError("An audio file is required")
        voice = await asyncio.to_thread(voice_library.add_voice, name, temp_path, audio.filename)
    except ValueError as exc:
        return _openai_error(str(exc), status_code=400)
    except Exception as exc:  # pragma: no cover - integration path
        logger.exception("Saved voice creation failed")
        return _openai_error(str(exc), status_code=500)
    finally:
        _cleanup_temp_files([temp_path])

    return JSONResponse(status_code=201, content=voice)


@app.patch("/v1/voices/{voice_id}", response_model=None)
async def rename_voice(voice_id: str, request: VoiceRenameRequest) -> JSONResponse:
    try:
        voice = await asyncio.to_thread(voice_library.rename_voice, voice_id, request.name)
    except FileNotFoundError as exc:
        return _openai_error(str(exc), status_code=404)
    except ValueError as exc:
        return _openai_error(str(exc), status_code=400)
    except Exception as exc:  # pragma: no cover - integration path
        logger.exception("Saved voice rename failed")
        return _openai_error(str(exc), status_code=500)

    return JSONResponse(content=voice)


@app.delete("/v1/voices/{voice_id}", response_model=None)
async def delete_voice(voice_id: str) -> JSONResponse:
    try:
        await asyncio.to_thread(voice_library.delete_voice, voice_id)
    except FileNotFoundError as exc:
        return _openai_error(str(exc), status_code=404)
    except Exception as exc:  # pragma: no cover - integration path
        logger.exception("Saved voice delete failed")
        return _openai_error(str(exc), status_code=500)

    return JSONResponse(content={"id": voice_id, "deleted": True})


@app.post("/v1/audio/speech", response_model=None)
async def create_speech(request: SpeechRequest) -> Response:
    try:
        audio_bytes, media_type = await asyncio.to_thread(service.synthesize, request)
    except ValueError as exc:
        return _openai_error(str(exc), status_code=400)
    except FileNotFoundError as exc:
        return _openai_error(str(exc), status_code=404)
    except Exception as exc:  # pragma: no cover - integration path
        logger.exception("Speech generation failed")
        return _openai_error(str(exc), status_code=500)

    return Response(content=audio_bytes, media_type=media_type)


@app.post("/ui/api/generate", response_model=None, include_in_schema=False)
async def create_speech_from_form(
    model: str = Form(default="voxcpm-tts"),
    input_text: str = Form(..., alias="input"),
    control: str | None = Form(default=None),
    saved_voice: str | None = Form(default=None),
    response_format: str = Form(default="wav"),
    speed: float = Form(default=1.0),
    cfg_value: float | None = Form(default=None),
    inference_timesteps: int | None = Form(default=None),
    max_len: int | None = Form(default=None),
    normalize: str | None = Form(default=None),
    denoise: str | None = Form(default=None),
    prompt_text: str | None = Form(default=None),
    prompt_audio: UploadFile | None = File(default=None),
    reference_audio: UploadFile | None = File(default=None),
) -> Response:
    temp_paths: list[str | None] = []
    try:
        prompt_audio_path = await _save_upload(prompt_audio)
        reference_audio_path = await _save_upload(reference_audio)
        temp_paths.extend([prompt_audio_path, reference_audio_path])

        request = SpeechRequest(
            model=model,
            input=input_text,
            control=(control or "").strip() or None,
            saved_voice=(saved_voice or "").strip() or None,
            response_format=response_format,
            speed=speed,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            normalize=_parse_bool_form(normalize),
            denoise=_parse_bool_form(denoise),
            prompt_text=(prompt_text or "").strip() or None,
            prompt_audio_path=prompt_audio_path,
            reference_audio_path=reference_audio_path,
        )
        audio_bytes, media_type = await asyncio.to_thread(service.synthesize, request)
    except ValueError as exc:
        return _openai_error(str(exc), status_code=400)
    except FileNotFoundError as exc:
        return _openai_error(str(exc), status_code=404)
    except Exception as exc:  # pragma: no cover - integration path
        logger.exception("Form speech generation failed")
        return _openai_error(str(exc), status_code=500)
    finally:
        _cleanup_temp_files(temp_paths)

    extension = {
        "audio/wav": "wav",
        "audio/flac": "flac",
        "audio/mpeg": "mp3",
        "application/octet-stream": "pcm",
    }.get(media_type, "bin")
    headers = {"Content-Disposition": f'inline; filename="voxcpm-output.{extension}"'}
    return Response(content=audio_bytes, media_type=media_type, headers=headers)


@app.get("/ui/voices", include_in_schema=False)
async def ui_voices() -> FileResponse:
    return FileResponse(STATIC_DIR / "voices.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve VoxCPM behind an OpenAI-compatible audio endpoint")
    parser.add_argument("--host", default=SETTINGS.host)
    parser.add_argument("--port", type=int, default=SETTINGS.port)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level=os.getenv("LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
