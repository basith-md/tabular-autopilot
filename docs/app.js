/* tabular-autopilot browser demo
 *
 * Boots Pyodide (CPython -> WebAssembly) in the background as soon as the
 * page loads, installs the real `tabular_autopilot` wheel via micropip, and
 * runs the exact same run_pipeline()/render_html() the CLI and Streamlit app
 * use -- entirely on the visitor's CPU. No file is ever sent anywhere.
 */

const WHEEL_URL = new URL("dist/tabular_autopilot-0.1.0-py3-none-any.whl", document.baseURI).href;
const SAMPLE_DIR = "samples/";
const MAX_ROWS_IN_BROWSER = 5000;

const statusEl = document.getElementById("engine-status");
const statusLabel = statusEl.querySelector(".label");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const stepsEl = document.getElementById("steps");

let pyodide = null;
let engineReady = false;
let pendingAction = null; // queued { kind: 'file'|'sample', ... } while engine still booting

function setStatus(kind, text) {
  statusEl.className = "engine-status" + (kind ? " " + kind : "");
  statusLabel.textContent = text;
}

async function loadPackageWithFallback(pyodideInstance, names) {
  try {
    await pyodideInstance.loadPackage(names);
  } catch (err) {
    console.warn(`pyodide.loadPackage(${names}) failed, falling back to micropip`, err);
    const micropip = pyodideInstance.pyimport("micropip");
    for (const name of names) {
      await micropip.install(name);
    }
  }
}

async function boot() {
  try {
    setStatus("", "Booting Python engine…");
    pyodide = await loadPyodide();

    setStatus("", "Loading numpy + pandas…");
    await loadPackageWithFallback(pyodide, ["numpy", "pandas"]);

    setStatus("", "Loading scipy + scikit-learn…");
    await loadPackageWithFallback(pyodide, ["scipy", "scikit-learn"]);

    setStatus("", "Loading matplotlib…");
    await loadPackageWithFallback(pyodide, ["matplotlib"]);

    setStatus("", "Loading statsmodels…");
    await loadPackageWithFallback(pyodide, ["statsmodels"]);

    setStatus("", "Loading micropip…");
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");

    setStatus("", "Installing seaborn, jinja2, openpyxl…");
    await micropip.install(["seaborn", "jinja2", "openpyxl"]);

    setStatus("", "Installing tabular-autopilot…");
    // deps=False: we've already loaded compatible versions of every
    // dependency above. Pyodide's prebuilt packages pin exact versions
    // (e.g. matplotlib 3.5.2) that can be older than this wheel's ">="
    // floors, which would otherwise make micropip's resolver refuse to
    // proceed even though everything it needs is already present.
    await micropip.install.callKwargs(WHEEL_URL, { deps: false });

    setStatus("", "Preparing analysis functions…");
    pyodide.runPython(PY_GLUE);

    engineReady = true;
    setStatus("ready", "Engine ready");

    if (pendingAction) {
      const action = pendingAction;
      pendingAction = null;
      runAction(action);
    }
  } catch (err) {
    console.error("Engine boot failed:", err);
    setStatus("error", "Engine failed to load — see console");
    renderFatalError(err);
  }
}

const PY_GLUE = `
import json
import traceback
import pandas as pd

def _read_any(path):
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def js_peek_columns(path):
    try:
        df = _read_any(path)
        return json.dumps({
            "ok": True,
            "columns": list(map(str, df.columns)),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
        })
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

def js_analyze(path, target, dataset_name, max_rows):
    try:
        from tabular_autopilot.pipeline import run_pipeline
        from tabular_autopilot.report import render_html

        df = _read_any(path)
        sampled = False
        if max_rows and len(df) > max_rows:
            df = df.sample(int(max_rows), random_state=42).reset_index(drop=True)
            sampled = True

        target = target if target else None
        result = run_pipeline(df, target=target, dataset_name=dataset_name)
        html = render_html(result)
        return json.dumps({
            "ok": True,
            "html": html,
            "sampled": sampled,
            "task": result.schema.task,
        })
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()})
`;

function renderFatalError(err) {
  stepsEl.innerHTML = `
    <div class="notice error">
      The in-browser Python engine couldn't finish loading (${escapeHtml(String(err))}).
      This can happen on older browsers or slow connections. You can still run the exact
      same analysis via the CLI or Streamlit app — see the
      <a href="https://github.com/basith-md/tabular-autopilot#quickstart">quickstart</a>.
    </div>`;
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/* ---------- File handling ---------- */

async function writeFileToFS(name, arrayBuffer) {
  const safeName = "/tmp/" + name.replace(/[^a-zA-Z0-9_.-]/g, "_");
  pyodide.FS.writeFile(safeName, new Uint8Array(arrayBuffer));
  return safeName;
}

function handleIncomingFile(file) {
  const action = { kind: "file", file };
  if (!engineReady) {
    pendingAction = action;
    renderWaitingForEngine(file.name);
    return;
  }
  runAction(action);
}

function handleSampleClick(name, target) {
  const action = { kind: "sample", name, target };
  if (!engineReady) {
    pendingAction = action;
    renderWaitingForEngine(name);
    return;
  }
  runAction(action);
}

function renderWaitingForEngine(label) {
  stepsEl.innerHTML = `
    <div class="step">
      <div class="analyzing">
        <div class="spinner"></div>
        <span>Engine is still warming up — <strong>${escapeHtml(label)}</strong> will be analyzed automatically the moment it's ready.</span>
      </div>
    </div>`;
}

async function runAction(action) {
  if (action.kind === "sample") {
    stepsEl.innerHTML = `<div class="step"><div class="analyzing"><div class="spinner"></div><span>Fetching sample dataset…</span></div></div>`;
    try {
      const resp = await fetch(SAMPLE_DIR + action.name);
      const buf = await resp.arrayBuffer();
      const path = await writeFileToFS(action.name, buf);
      await presentColumnPicker(path, action.name, buf.byteLength, action.target);
    } catch (err) {
      renderStepError("Could not load the sample dataset", err);
    }
    return;
  }

  // kind === "file"
  const file = action.file;
  try {
    const buf = await file.arrayBuffer();
    const path = await writeFileToFS(file.name, buf);
    await presentColumnPicker(path, file.name, buf.byteLength, null);
  } catch (err) {
    renderStepError("Could not read that file", err);
  }
}

function renderStepError(message, err) {
  console.error(message, err);
  stepsEl.innerHTML = `<div class="notice error">${escapeHtml(message)}: ${escapeHtml(String(err))}</div>`;
}

/* ---------- Column picker step ---------- */

async function presentColumnPicker(path, displayName, byteSize, presetTarget) {
  const peekFn = pyodide.globals.get("js_peek_columns");
  const raw = peekFn(path);
  const parsed = JSON.parse(raw);

  if (!parsed.ok) {
    renderStepError(`Couldn't parse ${displayName}`, parsed.error);
    return;
  }

  const options = ['<option value="">(none — profiling &amp; EDA only)</option>']
    .concat(parsed.columns.map((c) => {
      const selected = c === presetTarget ? "selected" : "";
      return `<option value="${escapeHtml(c)}" ${selected}>${escapeHtml(c)}</option>`;
    }))
    .join("");

  stepsEl.innerHTML = `
    <div class="step">
      <div class="step-label">1 · File loaded</div>
      <div class="file-summary">
        <span class="fname">${escapeHtml(displayName)}</span>
        <span class="fmeta">${parsed.n_rows.toLocaleString()} rows × ${parsed.n_cols} columns · ${fmtBytes(byteSize)}</span>
      </div>
      <div class="step-label">2 · Choose a target column (optional)</div>
      <div class="field-row">
        <select id="target-select">${options}</select>
        <button class="btn btn-primary" id="analyze-btn">Analyze ↦</button>
      </div>
      ${parsed.n_rows > MAX_ROWS_IN_BROWSER
        ? `<div class="notice warn">This dataset has ${parsed.n_rows.toLocaleString()} rows — for this browser demo it'll be randomly sampled down to ${MAX_ROWS_IN_BROWSER.toLocaleString()} to keep things responsive. The CLI and Streamlit app analyze the full dataset.</div>`
        : ""}
    </div>`;

  document.getElementById("analyze-btn").addEventListener("click", () => {
    const target = document.getElementById("target-select").value;
    runAnalysis(path, target, displayName);
  });
}

/* ---------- Analysis + results ---------- */

async function runAnalysis(path, target, displayName) {
  stepsEl.insertAdjacentHTML("beforeend", `
    <div class="step" id="analyze-progress">
      <div class="analyzing">
        <div class="spinner"></div>
        <span>Analyzing — schema inference, cleaning, feature engineering, model comparison, diagnostics… larger datasets can take up to a minute in-browser.</span>
      </div>
    </div>`);

  // Let the UI paint the spinner before the (blocking, synchronous) Python run starts.
  await new Promise((r) => setTimeout(r, 30));

  try {
    const analyzeFn = pyodide.globals.get("js_analyze");
    const raw = analyzeFn(path, target || "", displayName.replace(/\.[^.]+$/, ""), MAX_ROWS_IN_BROWSER);
    const parsed = JSON.parse(raw);

    const progressEl = document.getElementById("analyze-progress");
    if (progressEl) progressEl.remove();

    if (!parsed.ok) {
      renderStepError("Analysis failed", parsed.error);
      console.error(parsed.trace);
      return;
    }

    const sampledNote = parsed.sampled
      ? `<div class="notice info">Analyzed on a random 5,000-row sample of this dataset (browser-demo limit).</div>`
      : "";

    stepsEl.insertAdjacentHTML("beforeend", `
      <div class="step">
        <div class="step-label">3 · Result</div>
        ${sampledNote}
        <div class="result-frame-wrap">
          <iframe id="result-iframe" sandbox="allow-same-origin"></iframe>
        </div>
        <div style="margin-top:1rem;">
          <button class="btn btn-ghost" id="reset-btn">Analyze another file</button>
        </div>
      </div>`);

    const iframe = document.getElementById("result-iframe");
    iframe.addEventListener("load", () => {
      try {
        const h = iframe.contentDocument.documentElement.scrollHeight;
        iframe.style.height = Math.max(600, h + 40) + "px";
      } catch (e) {
        console.warn("Could not auto-size result iframe:", e);
      }
    });
    iframe.srcdoc = parsed.html;
    document.getElementById("reset-btn").addEventListener("click", () => {
      stepsEl.innerHTML = "";
    });
    iframe.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    const progressEl = document.getElementById("analyze-progress");
    if (progressEl) progressEl.remove();
    renderStepError("Analysis crashed", err);
  }
}

/* ---------- Wiring ---------- */

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleIncomingFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleIncomingFile(file);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    handleSampleClick(chip.dataset.sample, chip.dataset.target);
  });
});

boot();
