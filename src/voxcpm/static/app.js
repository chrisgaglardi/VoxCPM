const form = document.getElementById("tts-form");
const modelSelect = document.getElementById("model");
const resultStatus = document.getElementById("result-status");
const audioPlayer = document.getElementById("audio-player");
const downloadLink = document.getElementById("download-link");
const generateButton = document.getElementById("generate-button");
const modeIndicator = document.getElementById("mode-indicator");
const healthDot = document.getElementById("health-dot");
const healthText = document.getElementById("health-text");
const savedVoiceSelect = document.getElementById("saved_voice");

const EXAMPLES = {
  narrator: {
    input: "VoxCPM now serves both a browser workspace and an OpenAI-compatible endpoint from one deployment.",
    control: "Clear documentary narrator, calm and articulate, measured tempo.",
    prompt_text: "",
  },
  clone: {
    input: "This take uses your uploaded reference audio as the timbre anchor while preserving expressive control.",
    control: "Warm, slightly smiling, polished studio delivery.",
    prompt_text: "",
  },
  dialect: {
    input: "伙計，唔該一個A餐，凍奶茶少甜。",
    control: "粤语，中年男性，语气平淡，贴近日常对话。",
    prompt_text: "",
  },
};

function setHealth(state, text) {
  healthDot.classList.remove("ready", "error");
  if (state === "ready") {
    healthDot.classList.add("ready");
  }
  if (state === "error") {
    healthDot.classList.add("error");
  }
  healthText.textContent = text;
}

async function bootstrap() {
  try {
    const [healthRes, modelsRes, voicesRes] = await Promise.all([
      fetch("/healthz"),
      fetch("/v1/models"),
      fetch("/v1/voices"),
    ]);

    if (!healthRes.ok || !modelsRes.ok || !voicesRes.ok) {
      throw new Error("Server bootstrap failed");
    }

    const health = await healthRes.json();
    const models = await modelsRes.json();
    const voices = await voicesRes.json();
    const options = models.data || [];
    const savedVoices = voices.data || [];

    modelSelect.innerHTML = "";
    for (const item of options) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.id;
      modelSelect.appendChild(option);
    }

    savedVoiceSelect.innerHTML = "";
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = savedVoices.length ? "No saved voice selected" : "No saved voices yet";
    savedVoiceSelect.appendChild(emptyOption);

    for (const voice of savedVoices) {
      const option = document.createElement("option");
      option.value = voice.name;
      option.textContent = voice.name;
      savedVoiceSelect.appendChild(option);
    }

    setHealth("ready", health.model_loaded ? "Server ready. Model is warm." : "Server ready. Model loads on first request.");
  } catch (error) {
    console.error(error);
    setHealth("error", "Could not reach the server.");
    resultStatus.textContent = "The frontend loaded, but the API is not reachable yet.";
  }
}

function updateModeIndicator() {
  const hasReference = document.getElementById("reference_audio").files.length > 0;
  const hasSavedVoice = savedVoiceSelect.value.trim().length > 0;
  const hasPrompt = document.getElementById("prompt_audio").files.length > 0;
  const hasPromptText = document.getElementById("prompt_text").value.trim().length > 0;

  if (hasPrompt && hasPromptText) {
    modeIndicator.textContent = "Mode: ultimate cloning";
    return;
  }
  if (hasReference || hasSavedVoice) {
    modeIndicator.textContent = "Mode: controllable cloning";
    return;
  }
  modeIndicator.textContent = "Mode: voice design";
}

function setBusy(isBusy, message) {
  generateButton.disabled = isBusy;
  generateButton.textContent = isBusy ? "Generating…" : "Generate Audio";
  resultStatus.textContent = message;
}

document.querySelectorAll(".example-chip").forEach((button) => {
  button.addEventListener("click", () => {
    const example = EXAMPLES[button.dataset.example];
    if (!example) return;
    document.getElementById("input").value = example.input;
    document.getElementById("control").value = example.control;
    document.getElementById("prompt_text").value = example.prompt_text;
    updateModeIndicator();
  });
});

["reference_audio", "prompt_audio", "prompt_text", "saved_voice"].forEach((id) => {
  document.getElementById(id).addEventListener("change", updateModeIndicator);
  document.getElementById(id).addEventListener("input", updateModeIndicator);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  formData.set("normalize", document.getElementById("normalize").checked ? "true" : "false");
  formData.set("denoise", document.getElementById("denoise").checked ? "true" : "false");
  const hasReferenceUpload = document.getElementById("reference_audio").files.length > 0;
  const hasSavedVoice = savedVoiceSelect.value.trim().length > 0;

  if (hasReferenceUpload && hasSavedVoice) {
    resultStatus.textContent = "Choose either a saved voice or a direct reference upload, not both.";
    return;
  }

  setBusy(true, "Uploading inputs and generating audio...");
  downloadLink.classList.add("disabled");
  downloadLink.removeAttribute("href");

  try {
    const response = await fetch("/ui/api/generate", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const message = payload?.error?.message || "Generation failed.";
      throw new Error(message);
    }

    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    audioPlayer.src = blobUrl;
    audioPlayer.load();

    const extension = (response.headers.get("content-disposition") || "").match(/\.([a-z0-9]+)"/i)?.[1]
      || document.getElementById("response_format").value;
    downloadLink.href = blobUrl;
    downloadLink.download = `voxcpm-output.${extension}`;
    downloadLink.classList.remove("disabled");

    resultStatus.textContent = `Done. ${blob.size.toLocaleString()} bytes returned as ${extension.toUpperCase()}.`;
  } catch (error) {
    console.error(error);
    resultStatus.textContent = error.message;
  } finally {
    setBusy(false, resultStatus.textContent);
  }
});

bootstrap();
updateModeIndicator();
