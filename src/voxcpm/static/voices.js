const voiceUploadForm = document.getElementById("voice-upload-form");
const voiceUploadButton = document.getElementById("voice-upload-button");
const voiceList = document.getElementById("voice-list");
const libraryStatus = document.getElementById("library-status");
const voiceCount = document.getElementById("voice-count");

function setLibraryStatus(message) {
  libraryStatus.textContent = message;
}

function setUploadBusy(isBusy) {
  voiceUploadButton.disabled = isBusy;
  voiceUploadButton.textContent = isBusy ? "Saving…" : "Save Voice";
}

function renderVoices(voices) {
  voiceCount.textContent = `${voices.length} ${voices.length === 1 ? "voice" : "voices"}`;
  voiceList.innerHTML = "";

  if (!voices.length) {
    const emptyState = document.createElement("p");
    emptyState.className = "empty-state";
    emptyState.textContent = "No saved voices yet. Upload one to make it reusable from the studio or API.";
    voiceList.appendChild(emptyState);
    return;
  }

  for (const voice of voices) {
    const row = document.createElement("article");
    row.className = "voice-row";
    row.dataset.voiceId = voice.id;

    const meta = document.createElement("div");
    meta.className = "voice-meta";

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.value = voice.name;
    nameInput.maxLength = 80;
    nameInput.className = "voice-name-input";

    const detail = document.createElement("p");
    detail.className = "voice-detail";
    detail.textContent = `Saved name for API calls: ${voice.name}`;

    meta.appendChild(nameInput);
    meta.appendChild(detail);

    const actions = document.createElement("div");
    actions.className = "voice-actions";

    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "action-secondary";
    renameButton.textContent = "Rename";
    renameButton.addEventListener("click", async () => {
      const nextName = nameInput.value.trim();
      if (!nextName) {
        setLibraryStatus("Voice name cannot be empty.");
        return;
      }
      renameButton.disabled = true;
      try {
        const response = await fetch(`/v1/voices/${voice.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: nextName }),
        });
        const payload = await response.json().catch(() => null);
        if (!response.ok) {
          throw new Error(payload?.error?.message || "Rename failed.");
        }
        setLibraryStatus(`Renamed voice to ${payload.name}.`);
        await loadVoices();
      } catch (error) {
        console.error(error);
        setLibraryStatus(error.message);
      } finally {
        renameButton.disabled = false;
      }
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "action-secondary action-danger";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", async () => {
      const confirmed = window.confirm(`Delete saved voice "${voice.name}"?`);
      if (!confirmed) {
        return;
      }
      deleteButton.disabled = true;
      try {
        const response = await fetch(`/v1/voices/${voice.id}`, { method: "DELETE" });
        const payload = await response.json().catch(() => null);
        if (!response.ok) {
          throw new Error(payload?.error?.message || "Delete failed.");
        }
        setLibraryStatus(`Deleted voice ${voice.name}.`);
        await loadVoices();
      } catch (error) {
        console.error(error);
        setLibraryStatus(error.message);
      } finally {
        deleteButton.disabled = false;
      }
    });

    actions.appendChild(renameButton);
    actions.appendChild(deleteButton);
    row.appendChild(meta);
    row.appendChild(actions);
    voiceList.appendChild(row);
  }
}

async function loadVoices() {
  try {
    const response = await fetch("/v1/voices");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.error?.message || "Failed to load voices.");
    }
    renderVoices(payload.data || []);
    setLibraryStatus("Ready. Manage saved voices below.");
  } catch (error) {
    console.error(error);
    setLibraryStatus(error.message);
  }
}

voiceUploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(voiceUploadForm);
  if (!formData.get("name") || !formData.get("audio")) {
    setLibraryStatus("Provide both a voice name and an audio file.");
    return;
  }

  setUploadBusy(true);
  setLibraryStatus("Uploading and saving voice...");

  try {
    const response = await fetch("/v1/voices", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.error?.message || "Upload failed.");
    }
    voiceUploadForm.reset();
    setLibraryStatus(`Saved voice ${payload.name}.`);
    await loadVoices();
  } catch (error) {
    console.error(error);
    setLibraryStatus(error.message);
  } finally {
    setUploadBusy(false);
  }
});

loadVoices();
