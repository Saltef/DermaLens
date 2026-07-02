const input = document.querySelector("#photo-input");
const button = document.querySelector("#analyze-button");
const preview = document.querySelector("#preview");
const emptyState = document.querySelector("#empty-state");
const runState = document.querySelector("#run-state");
const findingsList = document.querySelector("#findings-list");
const qualityPanel = document.querySelector("#quality-panel");
const privacyPanel = document.querySelector("#privacy-panel");
const limitationsPanel = document.querySelector("#limitations-panel");
const topCard = document.querySelector("#top-card");
const summaryLine = document.querySelector("#summary-line");
const modelMode = document.querySelector("#model-mode");
const calibrationState = document.querySelector("#calibration-state");

let selectedFile = null;
let previewUrl = null;

input.addEventListener("change", () => {
  selectedFile = input.files?.[0] ?? null;
  button.disabled = !selectedFile;
  clearResults();
  runState.textContent = selectedFile ? "Ready" : "Idle";
  summaryLine.textContent = selectedFile ? selectedFile.name : "Awaiting image";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  if (!selectedFile) {
    preview.hidden = true;
    emptyState.hidden = false;
    return;
  }

  previewUrl = URL.createObjectURL(selectedFile);
  preview.src = previewUrl;
  preview.hidden = false;
  emptyState.hidden = true;
});

button.addEventListener("click", async () => {
  if (!selectedFile) return;

  runState.textContent = "Analyzing";
  summaryLine.textContent = "Processing locally";
  button.disabled = true;
  clearResults();

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Request failed with ${response.status}`);
    }

    const payload = await response.json();
    renderResult(payload);
    runState.textContent = "Complete";
  } catch (error) {
    findingsList.innerHTML = `<div class="finding"><p class="warning-text">${escapeHtml(error.message)}</p></div>`;
    runState.textContent = "Error";
    summaryLine.textContent = "Run failed";
  } finally {
    button.disabled = false;
  }
});

function renderResult(payload) {
  const findings = payload.findings ?? [];
  const topFinding = findings[0];
  renderModel(payload.model);
  renderTopFinding(topFinding);
  renderFindings(findings);
  renderQuality(payload);
  renderPrivacy(payload);
  renderLimitations(payload);
  summaryLine.textContent = topFinding
    ? `${topFinding.label} is the highest-ranked signal`
    : "No model findings returned";
}

function renderModel(model) {
  const calibration = model.calibration ?? {};
  modelMode.textContent = model.source === "onnx" ? "Local ONNX" : "Heuristic";
  calibrationState.textContent = calibration.enabled
    ? `${calibration.profile} ${Number(calibration.alpha).toFixed(1)}`
    : "Off";
}

function renderTopFinding(finding) {
  if (!finding) {
    topCard.hidden = true;
    topCard.innerHTML = "";
    return;
  }
  const score = Math.round(finding.score * 100);
  topCard.hidden = false;
  topCard.innerHTML = `
    <h3>Highest-ranked signal</h3>
    <div class="finding-top">
      <strong>${escapeHtml(finding.label)}</strong>
      <span class="score">${score}%</span>
    </div>
    <div class="bar" aria-hidden="true"><span style="width:${score}%"></span></div>
    <p>${escapeHtml(finding.rationale)}</p>
  `;
}

function renderFindings(findings) {
  findingsList.innerHTML = findings
    .map((finding) => {
      const score = Math.round(finding.score * 100);
      return `
        <article class="finding">
          <div class="finding-top">
            <h3>${escapeHtml(finding.label)} <span class="level">${escapeHtml(finding.level)}</span></h3>
            <span class="score">${score}%</span>
          </div>
          <div class="bar" aria-hidden="true"><span style="width:${score}%"></span></div>
          <p>${escapeHtml(finding.rationale)}</p>
        </article>
      `;
    })
    .join("");
}

function renderQuality(payload) {
  const warnings = payload.quality.warnings.length
    ? `<ul>${payload.quality.warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : "<p>Image quality checks passed.</p>";

  const flags = payload.review_flags.length
    ? `<ul>${payload.review_flags.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : "<p>No additional review flags from this run.</p>";

  qualityPanel.innerHTML = `
    <h3>Model</h3>
    <p>${escapeHtml(payload.model.name)} (${escapeHtml(payload.model.source)})</p>
    <h3>Image Quality</h3>
    <p>Brightness ${payload.quality.brightness}, contrast ${payload.quality.contrast}</p>
    ${warnings}
    <h3>Review Flags</h3>
    ${flags}
  `;
}

function renderPrivacy(payload) {
  const shortHash = String(payload.privacy.sha256 ?? "").slice(0, 16);
  privacyPanel.innerHTML = `
    <h3>Privacy</h3>
    <p>EXIF stripped: ${payload.privacy.exif_stripped ? "yes" : "no"}</p>
    <p>Saved to disk: ${payload.privacy.image_saved ? "yes" : "no"}</p>
    <p class="hash">SHA-256: ${escapeHtml(shortHash)}...</p>
  `;
}

function renderLimitations(payload) {
  limitationsPanel.innerHTML = `
    <h3>Limitations</h3>
    <ul>${payload.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
  `;
}

function clearResults() {
  findingsList.innerHTML = "";
  qualityPanel.innerHTML = "";
  privacyPanel.innerHTML = "";
  limitationsPanel.innerHTML = "";
  topCard.hidden = true;
  topCard.innerHTML = "";
  calibrationState.textContent = "Pending";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
