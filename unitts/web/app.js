"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  providers: [],
  busy: false,
  connected: false,
  takes: [],
  active: null,
  context: null,
  voiceRequest: 0,
};
const defaults = {
  piper: ["en_US-lessac-medium", "en"],
  kokoro: ["af_heart", "en"],
  espeak: ["", "en"],
  openai: ["alloy", "en"],
  elevenlabs: ["", "en"],
  azure: ["en-US-JennyNeural", "en-US"],
};
const names = {
  piper: "Piper",
  espeak: "eSpeak NG",
  kokoro: "Kokoro",
  openai: "OpenAI",
  elevenlabs: "ElevenLabs",
  fish: "Fish Audio",
  smallest: "Smallest AI",
  azure: "Azure Speech",
};
const mimes = {
  wav: "audio/wav",
  mp3: "audio/mpeg",
  ogg: "audio/ogg",
  opus: "audio/ogg",
  flac: "audio/flac",
};
const examples = {
  welcome:
    "Hello, and welcome. Make yourself comfortable. Every good conversation starts with a voice, and this one starts with yours.",
  story:
    "The little bookshop stood at the end of a quiet street. Every morning, its owner opened the door, set a fresh cup of tea on the counter, and waited for a new story to begin.",
};
const selected = () =>
  state.providers.find((item) => item.name === $("provider").value);
const clock = (value) =>
  `${Math.floor((value || 0) / 60)}:${String(Math.floor((value || 0) % 60)).padStart(2, "0")}`;
function textState() {
  const text = $("text").value;
  $("char-count").textContent = `${text.length.toLocaleString()} characters`;
  $("word-count").textContent =
    `${text.trim() ? text.trim().split(/\s+/).length : 0} words`;
  $("synthesize").disabled = state.busy || !state.connected || !text.trim();
}
function providerState() {
  const item = selected();
  if (!item) return;
  const cap = item.capabilities;
  const features = [
    cap.provider_type === "api"
      ? "Hosted API"
      : cap.provider_type.toUpperCase(),
    ...(cap.voice_cloning ? ["Voice cloning"] : []),
    ...(cap.speed ? ["Speed control"] : []),
  ];
  $("provider-meta").replaceChildren(
    ...features.map((label) => {
      const span = document.createElement("span");
      span.textContent = label;
      return span;
    }),
  );
  $("format").replaceChildren(
    ...cap.output_formats.map(
      (format) => new Option(format.toUpperCase(), format),
    ),
  );
  $("format").value = cap.output_formats.includes("wav")
    ? "wav"
    : cap.output_formats[0];
  const preset = defaults[item.name] || ["", "en"];
  $("voice").value = preset[0];
  $("language").value = preset[1];
  $("model").value = "";
  $("voice-list").replaceChildren();
  $("voice-note").textContent = "Use the provider default or enter a voice ID.";
  $("speed").disabled = !cap.speed || item.name === "elevenlabs";
  $("speed").value = 1;
  $("speed-value").textContent = "1.00×";
  $("reference-section").hidden = !cap.voice_cloning || cap.provider_type === "api";
  $("reference-audio").value = "";
  $("reference-text").value = "";
  $("reference-section").open = false;
  $("device").value = "auto";
  for (const option of $("device").options)
    option.disabled =
      cap.provider_type === "api"
        ? option.value !== "auto"
        : cap.requires_gpu && option.value === "cpu";
  let notice = "";
  if (cap.provider_type === "api" && !item.configured)
    notice =
      "API key missing. Restart the server with --env-file .env after adding this provider's key.";
  else if (cap.requires_gpu)
    notice = `Requires a GPU${cap.min_gpu_vram_gb ? ` with approximately ${cap.min_gpu_vram_gb} GB VRAM` : ""} and the provider's model dependencies.`;
  else if (item.name === "elevenlabs")
    notice =
      "Speed control is unavailable in the current adapter. Account credits and model access are required.";
  $("provider-notice").textContent = notice;
  $("provider-notice").hidden = !notice;
  $("request-note").textContent =
    cap.provider_type === "api"
      ? "Hosted generation uses your provider's credits."
      : "First generation may download a voice. Following takes reuse the model.";
  state.voiceRequest++;
  $("load-voices").disabled = false;
  $("load-voices").textContent = "Browse voices ↗";
}
function busy(value) {
  state.busy = value;
  $("generate-label").textContent = value ? "Generating…" : "Generate speech";
  $("result-panel").classList.toggle("busy", value);
  $("result-panel").setAttribute("aria-busy", String(value));
  for (const id of [
    "provider",
    "voice",
    "language",
    "format",
    "device",
    "model",
    "reference-audio",
    "reference-text",
    "load-voices",
  ])
    $(id).disabled = value;
  $("speed").disabled =
    value ||
    !selected()?.capabilities.speed ||
    selected()?.name === "elevenlabs";
  textState();
}
function drawWave() {
  const canvas = $("waveform"),
    ctx = canvas.getContext("2d"),
    width = canvas.clientWidth,
    height = 43,
    scale = window.devicePixelRatio || 1;
  canvas.width = width * scale;
  canvas.height = height * scale;
  ctx.scale(scale, scale);
  const peaks = state.active?.peaks;
  if (!peaks || !width) return;
  const progress = $("audio").duration
      ? $("audio").currentTime / $("audio").duration
      : 0,
    bars = Math.max(1, Math.floor(width / 5));
  for (let i = 0; i < bars; i++) {
    const peak =
        peaks[
          Math.min(peaks.length - 1, Math.floor((i / bars) * peaks.length))
        ],
      barHeight = Math.max(3, peak * 40);
    ctx.fillStyle = i / bars < progress ? "#235e4c" : "#bdcdb1";
    ctx.fillRect(i * 5, (height - barHeight) / 2, 2.5, barHeight);
  }
}
function renderLibrary() {
  $("library-empty").hidden = state.takes.length > 0;
  $("library-count").textContent = state.takes.length;
  $("take-count").textContent =
    `${state.takes.length} take${state.takes.length === 1 ? "" : "s"}`;
  $("clear-library").disabled = !state.takes.length || state.busy;
  $("take-list").replaceChildren(
    ...state.takes.map((take) => {
      const row = document.createElement("div");
      row.className = `take-row${state.active === take ? " selected" : ""}`;
      const play = document.createElement("button");
      play.className = "take-load";
      play.textContent = "▶";
      play.type = "button";
      play.disabled = state.busy;
      play.setAttribute("aria-label", `Play take ${take.number}`);
      play.addEventListener("click", () => {
        activate(take);
        $("audio")
          .play()
          .catch(() => {});
      });
      const copy = document.createElement("div");
      copy.className = "take-copy";
      const title = document.createElement("strong");
      title.textContent = take.text;
      const meta = document.createElement("small");
      meta.textContent = `Take ${take.number} · ${names[take.provider] || take.provider} · ${take.format.toUpperCase()} · ${clock(take.duration)} · ${take.generated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
      copy.append(title, meta);
      const link = document.createElement("a");
      link.href = take.url;
      link.download = `unitts-take-${take.number}.${take.format}`;
      link.textContent = "↓";
      link.setAttribute("aria-label", `Download take ${take.number}`);
      row.append(play, copy, link);
      return row;
    }),
  );
}
function activate(take) {
  state.active = take;
  $("audio").src = take.url;
  $("playback").hidden = false;
  $("empty-output").hidden = true;
  $("download").hidden = false;
  $("download").href = take.url;
  $("download").download = `unitts-take-${take.number}.${take.format}`;
  $("download").textContent = `↓ Download ${take.format.toUpperCase()}`;
  $("result-title").textContent =
    `Take ${String(take.number).padStart(2, "0")} · ${names[take.provider] || take.provider}`;
  $("result-badge").textContent = "Ready to play";
  $("result-message").textContent =
    `${take.voice || "Default voice"} · ${Math.round(take.bytes / 1024)} KB${take.rate ? ` · ${take.rate.toLocaleString()} Hz` : ""}`;
  $("result-stats").textContent = `Generated in ${take.time.toFixed(1)}s`;
  $("duration").textContent = clock(take.duration);
  $("elapsed").textContent = "0:00";
  $("seek").value = 0;
  $("warnings").hidden = !take.warnings.length;
  $("warnings").replaceChildren(
    ...take.warnings.map((warning) => {
      const li = document.createElement("li");
      li.textContent = warning;
      return li;
    }),
  );
  $("result-panel").classList.remove("error");
  drawWave();
  renderLibrary();
}
async function analyze(bytes) {
  try {
    state.context ||= new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await state.context.decodeAudioData(bytes.buffer.slice(0));
    const samples = decoded.getChannelData(0),
      peaks = new Float32Array(240),
      step = Math.max(1, Math.floor(samples.length / peaks.length));
    for (let i = 0; i < peaks.length; i++)
      for (let j = i * step; j < Math.min(samples.length, (i + 1) * step); j++)
        peaks[i] = Math.max(peaks[i], Math.abs(samples[j]));
    const max = Math.max(...peaks, 0.001);
    for (let i = 0; i < peaks.length; i++) peaks[i] /= max;
    return { peaks, duration: decoded.duration, rate: decoded.sampleRate };
  } catch {
    return { peaks: null, duration: 0, rate: null };
  }
}
async function readReference(file) {
  if (!file) return null;
  if (file.size > 20 * 1024 * 1024)
    throw new Error("Reference audio must be smaller than 20 MB.");
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve({
        name: file.name,
        data: String(reader.result).split(",", 2)[1],
      });
    reader.onerror = () => reject(new Error("Could not read reference audio."));
    reader.readAsDataURL(file);
  });
}
async function generate() {
  if (state.busy || $("synthesize").disabled) return;
  const settings = {
    provider: $("provider").value,
    text: $("text").value.trim(),
    voice: $("voice").value,
    language: $("language").value,
    output_format: $("format").value,
    speed: Number($("speed").value),
    device: $("device").value,
    model: $("model").value,
    reference_text: $("reference-text").value,
  };
  busy(true);
  renderLibrary();
  $("audio").pause();
  $("result-panel").classList.remove("error");
  $("result-title").textContent = "Finding your voice…";
  $("result-badge").textContent = "Generating";
  $("result-message").textContent =
    "Your provider is working. First-time model downloads can take a few minutes.";
  $("result-stats").textContent = "";
  $("warnings").hidden = true;
  $("playback").hidden = true;
  $("empty-output").hidden = false;
  const started = performance.now();
  try {
    settings.reference_audio = await readReference(
      $("reference-audio").files[0],
    );
    const response = await fetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      }),
      payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Generation failed.");
    const bytes = Uint8Array.from(atob(payload.audio), (char) =>
      char.charCodeAt(0),
    );
    if (!bytes.length)
      throw new Error("The provider returned empty audio. Please try again.");
    const format = payload.output_format || settings.output_format,
      analysis = await analyze(bytes);
    const take = {
      ...settings,
      ...analysis,
      rate: payload.sample_rate || analysis.rate,
      number: (state.takes[0]?.number || 0) + 1,
      format,
      warnings: payload.warnings || [],
      bytes: bytes.length,
      time: (performance.now() - started) / 1000,
      generated: new Date(),
      url: URL.createObjectURL(new Blob([bytes], { type: mimes[format] })),
    };
    state.takes.unshift(take);
    if (state.takes.length > 20) URL.revokeObjectURL(state.takes.pop().url);
    activate(take);
  } catch (error) {
    $("result-panel").classList.add("error");
    $("result-title").textContent = "This take needs another try.";
    $("result-badge").textContent = "Generation failed";
    $("result-message").textContent =
      error.message ||
      "Something went wrong. Check your connection and provider setup.";
  } finally {
    busy(false);
    renderLibrary();
  }
}
async function loadVoices() {
  const request = ++state.voiceRequest,
    provider = $("provider").value;
  $("load-voices").disabled = true;
  $("load-voices").textContent = "Loading…";
  try {
    const response = await fetch(
        `/api/voices?provider=${encodeURIComponent(provider)}`,
      ),
      payload = await response.json();
    if (request !== state.voiceRequest) return;
    if (!response.ok)
      throw new Error(payload.error || "Unable to load voices.");
    $("voice-list").replaceChildren(
      ...payload.voices.map(
        (voice) =>
          new Option(
            typeof voice.name === "string" ? voice.name : voice.id,
            voice.id,
          ),
      ),
    );
    $("voice-note").textContent = payload.voices.length
      ? `${payload.voices.length} voices loaded. Click the voice field to choose.`
      : "This provider has no voice catalog. Enter its voice ID manually.";
  } catch (error) {
    if (request === state.voiceRequest)
      $("voice-note").textContent = error.message;
  } finally {
    if (request === state.voiceRequest) {
      $("load-voices").disabled = state.busy;
      $("load-voices").textContent = "Browse voices ↗";
    }
  }
}
async function initialize() {
  try {
    const response = await fetch("/api/providers"),
      payload = await response.json();
    if (!response.ok) throw new Error("Could not load providers.");
    state.providers = payload.providers;
    $("provider").replaceChildren(
      ...[
        ["cpu", "Local · CPU"],
        ["api", "Hosted APIs"],
        ["gpu", "Local · GPU"],
      ].map(([type, label]) => {
        const group = document.createElement("optgroup");
        group.label = label;
        for (const item of state.providers.filter(
          (item) => item.capabilities.provider_type === type,
        ))
          group.append(new Option(names[item.name] || item.name, item.name));
        return group;
      }),
    );
    $("provider").value = "piper";
    state.connected = true;
    providerState();
    textState();
    $("server-label").textContent = "Server connected";
  } catch (error) {
    $("server-state").classList.add("error");
    $("server-label").textContent = "Server unavailable";
    $("result-message").textContent =
      `${error.message} Restart the local server and reload.`;
  }
}
$("provider").addEventListener("change", providerState);
$("text").addEventListener("input", textState);
$("speed").addEventListener("input", () => {
  $("speed-value").textContent = `${Number($("speed").value).toFixed(2)}×`;
});
$("synthesize").addEventListener("click", generate);
$("load-voices").addEventListener("click", loadVoices);
$("clear-text").addEventListener("click", () => {
  $("text").value = "";
  textState();
  $("text").focus();
});
document.querySelectorAll("[data-example]").forEach((button) =>
  button.addEventListener("click", () => {
    $("text").value = examples[button.dataset.example];
    textState();
    $("text").focus();
  }),
);
$("import-text").addEventListener("click", () => $("text-file").click());
$("text-file").addEventListener("change", async () => {
  const file = $("text-file").files[0];
  if (!file) return;
  if (file.size > 1024 * 1024) {
    $("result-message").textContent = "Import a text file smaller than 1 MB.";
    return;
  }
  $("text").value = await file.text();
  $("text-file").value = "";
  textState();
});
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    generate();
  }
});
$("play-toggle").addEventListener("click", () => {
  if ($("audio").paused)
    $("audio")
      .play()
      .catch(() => {
        $("result-message").textContent =
          "Your browser cannot play this format. Download the file to listen.";
      });
  else $("audio").pause();
});
$("audio").addEventListener("play", () => {
  $("play-toggle").textContent = "Ⅱ";
  $("play-toggle").setAttribute("aria-label", "Pause audio");
});
$("audio").addEventListener("pause", () => {
  $("play-toggle").textContent = "▶";
  $("play-toggle").setAttribute("aria-label", "Play audio");
});
$("audio").addEventListener("loadedmetadata", () => {
  $("duration").textContent = clock($("audio").duration);
});
$("audio").addEventListener("timeupdate", () => {
  $("elapsed").textContent = clock($("audio").currentTime);
  $("seek").value = $("audio").duration
    ? ($("audio").currentTime / $("audio").duration) * 100
    : 0;
  drawWave();
});
$("seek").addEventListener("input", () => {
  if (Number.isFinite($("audio").duration))
    $("audio").currentTime =
      (Number($("seek").value) / 100) * $("audio").duration;
});
$("clear-library").addEventListener("click", () => {
  $("audio").pause();
  $("audio").removeAttribute("src");
  $("audio").load();
  state.takes.forEach((take) => URL.revokeObjectURL(take.url));
  state.takes = [];
  state.active = null;
  $("playback").hidden = true;
  $("empty-output").hidden = false;
  $("download").hidden = true;
  $("result-title").textContent = "Your next take starts here.";
  $("result-badge").textContent = "No audio yet";
  $("result-message").textContent =
    "Generate your first take to listen, fine-tune, and download.";
  $("result-stats").textContent = "";
  $("warnings").hidden = true;
  $("result-panel").classList.remove("error");
  renderLibrary();
});
$("help-open").addEventListener("click", () => $("help-dialog").showModal());
$("help-close").addEventListener("click", () => $("help-dialog").close());
new ResizeObserver(drawWave).observe($("waveform"));
initialize();
