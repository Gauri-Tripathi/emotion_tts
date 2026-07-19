const elements = {
  audio: document.querySelector("#audio"),
  charCount: document.querySelector("#char-count"),
  device: document.querySelector("#device"),
  download: document.querySelector("#download"),
  language: document.querySelector("#language"),
  model: document.querySelector("#model"),
  provider: document.querySelector("#provider"),
  providerMeta: document.querySelector("#provider-meta"),
  referenceAudio: document.querySelector("#reference-audio"),
  referenceText: document.querySelector("#reference-text"),
  requestNote: document.querySelector("#request-note"),
  resultMessage: document.querySelector("#result-message"),
  resultTitle: document.querySelector("#result-title"),
  serverState: document.querySelector("#server-state"),
  speed: document.querySelector("#speed"),
  speedValue: document.querySelector("#speed-value"),
  synthesize: document.querySelector("#synthesize"),
  text: document.querySelector("#text"),
  voice: document.querySelector("#voice"),
};

let providers = [];
let audioUrl = null;

function selectedProvider() {
  return providers.find((item) => item.name === elements.provider.value);
}

function updateProviderDetails() {
  const item = selectedProvider();
  if (!item) return;
  const cap = item.capabilities;
  const features = [
    cap.voice_cloning ? "Voice cloning" : null,
    cap.voice_description ? "Voice design" : null,
    cap.emotion ? "Emotion" : null,
    cap.streaming ? "Streaming" : null,
  ].filter(Boolean);
  elements.providerMeta.textContent = `${cap.provider_type.toUpperCase()} | ${cap.languages.join(", ") || "auto"}${features.length ? ` | ${features.join(", ")}` : ""}`;
  elements.requestNote.textContent = cap.notes || "Uses provider defaults for settings not shown here.";
  const supportsReference = cap.voice_cloning;
  elements.referenceAudio.disabled = !supportsReference;
  elements.referenceText.disabled = !supportsReference;
  if (!supportsReference) {
    elements.referenceAudio.value = "";
    elements.referenceText.value = "";
  }
}

function updateTextState() {
  const count = elements.text.value.length;
  elements.charCount.textContent = `${count} character${count === 1 ? "" : "s"}`;
  elements.synthesize.disabled = !count || !providers.length;
}

function setServerState(label, state) {
  elements.serverState.textContent = label;
  elements.serverState.className = `server-state ${state}`;
}

function setResult(title, message) {
  elements.resultTitle.textContent = title;
  elements.resultMessage.textContent = message;
}

function readReference(file) {
  if (!file) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({ name: file.name, data: String(reader.result).split(",", 2)[1] });
    reader.onerror = () => reject(new Error("Could not read reference audio"));
    reader.readAsDataURL(file);
  });
}

async function synthesize() {
  elements.synthesize.disabled = true;
  elements.synthesize.textContent = "Synthesizing";
  setResult("Synthesis in progress", "The selected provider is working. This may take a while for a local GPU model.");
  try {
    const referenceAudio = await readReference(elements.referenceAudio.files[0]);
    const response = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: elements.provider.value,
        text: elements.text.value,
        language: elements.language.value,
        voice: elements.voice.value,
        speed: Number(elements.speed.value),
        device: elements.device.value,
        model: elements.model.value,
        reference_audio: referenceAudio,
        reference_text: elements.referenceText.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Synthesis failed");
    const bytes = Uint8Array.from(atob(payload.audio), (character) => character.charCodeAt(0));
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    elements.audio.src = audioUrl;
    elements.download.href = audioUrl;
    elements.download.hidden = false;
    elements.audio.play().catch(() => {});
    const warnings = payload.warnings.length ? ` Warnings: ${payload.warnings.join("; ")}` : "";
    setResult("Audio ready", `${Math.round(bytes.length / 1024)} KB WAV at ${payload.sample_rate} Hz.${warnings}`);
  } catch (error) {
    setResult("Synthesis failed", error.message || "The provider returned an unknown error.");
  } finally {
    elements.synthesize.textContent = "Synthesize";
    updateTextState();
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/providers");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to load providers");
    providers = payload.providers;
    elements.provider.replaceChildren(...providers.map((item) => {
      const option = document.createElement("option");
      option.value = item.name;
      option.textContent = item.name;
      return option;
    }));
    updateProviderDetails();
    updateTextState();
    setServerState("Local server ready", "ready");
  } catch (error) {
    setServerState("Server unavailable", "error");
    setResult("Cannot connect", error.message || "Start the local UniTTS web server and reload this page.");
  }
}

elements.provider.addEventListener("change", updateProviderDetails);
elements.speed.addEventListener("input", () => { elements.speedValue.textContent = `${Number(elements.speed.value).toFixed(2)}x`; });
elements.text.addEventListener("input", updateTextState);
elements.synthesize.addEventListener("click", synthesize);
initialize();
