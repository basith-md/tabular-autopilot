/* Tabular Autopilot browser demo
 *
 * Boots Pyodide (CPython -> WebAssembly) in the background as soon as the
 * page loads, installs the real `tabular_autopilot` wheel via micropip, and
 * runs the exact same run_pipeline()/render_html() the CLI and Streamlit app
 * use -- entirely on the visitor's CPU. No file is ever sent anywhere.
 */

const WHEEL_URL = new URL("dist/tabular_autopilot-0.1.0-py3-none-any.whl", document.baseURI).href;
const SAMPLE_DIR = "samples/";

const statusEl = document.getElementById("engine-status");
const statusLabel = statusEl.querySelector(".label");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const stepsEl = document.getElementById("steps");

let pyodide = null;
let engineReady = false;
let pendingAction = null; // queued { kind: 'file'|'sample', ... } while engine still booting
let lastLoaded = null; // { path, displayName, byteSize } for "adjust settings" re-runs

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

    setStatus("", "Installing Tabular Autopilot…");
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

def js_peek_columns(path):
    try:
        from tabular_autopilot.pipeline import load_dataframe
        df = load_dataframe(path)
        return json.dumps({
            "ok": True,
            "columns": list(map(str, df.columns)),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
        })
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

def js_analyze(path, target, dataset_name, max_rows, test_size, impute_strategy, model_names_json,
               handle_imbalance, vectorize_text, feature_selection, cv_folds, cap_outliers,
               ridge_alpha_min, ridge_alpha_max, lasso_alpha_min, lasso_alpha_max, logreg_c,
               rf_n_estimators, rf_max_depth, gb_learning_rate, gb_max_iter, hyperparameter_search,
               broad_hyperparameter_search, high_cardinality_encoding, intervention_date):
    try:
        from tabular_autopilot.pipeline import load_dataframe, run_pipeline
        from tabular_autopilot.report import render_html

        model_names = json.loads(model_names_json) or None

        df = load_dataframe(path)
        sampled = False
        if max_rows and len(df) > max_rows:
            df = df.sample(int(max_rows), random_state=42).reset_index(drop=True)
            sampled = True

        target = target if target else None
        result = run_pipeline(
            df,
            target=target,
            dataset_name=dataset_name,
            numeric_impute_strategy=impute_strategy,
            test_size=float(test_size),
            model_names=model_names,
            handle_imbalance=bool(handle_imbalance),
            vectorize_text=bool(vectorize_text),
            feature_selection=bool(feature_selection),
            cv_folds=int(cv_folds),
            cap_outliers=bool(cap_outliers),
            ridge_alpha_range=(float(ridge_alpha_min), float(ridge_alpha_max)),
            lasso_alpha_range=(float(lasso_alpha_min), float(lasso_alpha_max)),
            logreg_C=float(logreg_c),
            rf_n_estimators=int(rf_n_estimators),
            rf_max_depth=int(rf_max_depth) if int(rf_max_depth) > 0 else None,
            gb_learning_rate=float(gb_learning_rate),
            gb_max_iter=int(gb_max_iter),
            hyperparameter_search=bool(hyperparameter_search),
            broad_hyperparameter_search=bool(broad_hyperparameter_search),
            high_cardinality_encoding=high_cardinality_encoding,
            intervention_date=intervention_date if intervention_date else None,
        )
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

/* ---------- Settings panel ---------- */

const settingsToggle = document.getElementById("settings-toggle");
const settingsPanel = document.getElementById("settings-panel");
settingsToggle.addEventListener("click", () => {
  settingsToggle.classList.toggle("open");
  settingsPanel.classList.toggle("open");
});

const testSizeInput = document.getElementById("test-size");
const testSizeValue = document.getElementById("test-size-value");
testSizeInput.addEventListener("input", () => {
  testSizeValue.textContent = `${testSizeInput.value}%`;
});

const maxRowsInput = document.getElementById("max-rows");
const maxRowsValue = document.getElementById("max-rows-value");
maxRowsInput.addEventListener("input", () => {
  maxRowsValue.textContent = Number(maxRowsInput.value).toLocaleString();
});

const useCvInput = document.getElementById("use-cv");
const cvFoldsRow = document.getElementById("cv-folds-row");
const cvFoldsInput = document.getElementById("cv-folds");
const cvFoldsValue = document.getElementById("cv-folds-value");
useCvInput.addEventListener("change", () => {
  cvFoldsRow.style.display = useCvInput.checked ? "grid" : "none";
});
cvFoldsInput.addEventListener("input", () => {
  cvFoldsValue.textContent = cvFoldsInput.value;
});

/* Model checkboxes reveal their own hyperparameter sub-panel, and only
   that panel -- customization is scoped to the model it belongs to. */
document.querySelectorAll("#model-checkboxes input[type=checkbox]").forEach((cb) => {
  const panel = document.querySelector(`.model-hp-panel[data-for-model="${cb.value}"]`);
  if (!panel) return;
  const sync = () => panel.classList.toggle("open", cb.checked);
  cb.addEventListener("change", sync);
  sync();
});

function wireRangeDisplay(inputId, valueId, format = (v) => v) {
  const input = document.getElementById(inputId);
  const value = document.getElementById(valueId);
  const render = () => { value.textContent = format(input.value); };
  input.addEventListener("input", render);
  render();
}
wireRangeDisplay("logreg-c", "logreg-c-value", (v) => Number(v).toFixed(2));
wireRangeDisplay("rf-n-estimators", "rf-n-estimators-value");
wireRangeDisplay("rf-max-depth", "rf-max-depth-value", (v) => (Number(v) === 0 ? "unlimited" : v));
wireRangeDisplay("gb-learning-rate", "gb-learning-rate-value", (v) => Number(v).toFixed(2));
wireRangeDisplay("gb-max-iter", "gb-max-iter-value");

function currentSettings() {
  const modelCheckboxes = [...document.querySelectorAll("#model-checkboxes input:checked")];
  return {
    testSize: Number(testSizeInput.value) / 100,
    maxRows: Number(maxRowsInput.value),
    imputeStrategy: document.querySelector('input[name="impute"]:checked').value,
    modelNames: modelCheckboxes.map((c) => c.value),
    handleImbalance: document.getElementById("handle-imbalance").checked,
    vectorizeText: document.getElementById("vectorize-text").checked,
    featureSelection: document.getElementById("feature-selection").checked,
    cvFolds: useCvInput.checked ? Number(cvFoldsInput.value) : 0,
    capOutliers: document.getElementById("cap-outliers").checked,
    hyperparameterSearch: document.getElementById("hyperparameter-search").checked,
    ridgeAlphaMin: Number(document.getElementById("ridge-alpha-min").value),
    ridgeAlphaMax: Number(document.getElementById("ridge-alpha-max").value),
    lassoAlphaMin: Number(document.getElementById("lasso-alpha-min").value),
    lassoAlphaMax: Number(document.getElementById("lasso-alpha-max").value),
    logregC: Number(document.getElementById("logreg-c").value),
    rfNEstimators: Number(document.getElementById("rf-n-estimators").value),
    rfMaxDepth: Number(document.getElementById("rf-max-depth").value),
    gbLearningRate: Number(document.getElementById("gb-learning-rate").value),
    gbMaxIter: Number(document.getElementById("gb-max-iter").value),
    broadHyperparameterSearch: document.getElementById("broad-hyperparameter-search").checked,
    highCardinalityEncoding: document.querySelector('input[name="high-card-encoding"]:checked').value,
    interventionDate: document.getElementById("intervention-date").value.trim(),
  };
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

  lastLoaded = { path, displayName, byteSize };

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
      ${parsed.n_rows > maxRowsInput.value
        ? `<div class="notice warn">This dataset has ${parsed.n_rows.toLocaleString()} rows — for this browser demo it'll be randomly sampled down per the row cap in Advanced settings (currently ${Number(maxRowsInput.value).toLocaleString()}) to keep things responsive. The CLI and Streamlit app analyze the full dataset.</div>`
        : ""}
    </div>`;

  document.getElementById("analyze-btn").addEventListener("click", () => {
    const target = document.getElementById("target-select").value;
    runAnalysis(path, target, displayName);
  });
}

/* ---------- Analysis + results ---------- */

async function runAnalysis(path, target, displayName) {
  const settings = currentSettings();

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
    const raw = analyzeFn(
      path,
      target || "",
      displayName.replace(/\.[^.]+$/, ""),
      settings.maxRows,
      settings.testSize,
      settings.imputeStrategy,
      JSON.stringify(settings.modelNames),
      settings.handleImbalance,
      settings.vectorizeText,
      settings.featureSelection,
      settings.cvFolds,
      settings.capOutliers,
      settings.ridgeAlphaMin,
      settings.ridgeAlphaMax,
      settings.lassoAlphaMin,
      settings.lassoAlphaMax,
      settings.logregC,
      settings.rfNEstimators,
      settings.rfMaxDepth,
      settings.gbLearningRate,
      settings.gbMaxIter,
      settings.hyperparameterSearch,
      settings.broadHyperparameterSearch,
      settings.highCardinalityEncoding,
      settings.interventionDate
    );
    const parsed = JSON.parse(raw);

    const progressEl = document.getElementById("analyze-progress");
    if (progressEl) progressEl.remove();

    if (!parsed.ok) {
      renderStepError("Analysis failed", parsed.error);
      console.error(parsed.trace);
      return;
    }

    const sampledNote = parsed.sampled
      ? `<div class="notice info">Analyzed on a random ${settings.maxRows.toLocaleString()}-row sample of this dataset (browser-demo limit — adjustable in Advanced settings).</div>`
      : "";

    stepsEl.insertAdjacentHTML("beforeend", `
      <div class="step">
        <div class="step-label">3 · Result</div>
        ${sampledNote}
        <div class="result-frame-wrap">
          <iframe id="result-iframe" sandbox="allow-same-origin"></iframe>
        </div>
        <div class="result-actions">
          <button class="btn btn-ghost" id="adjust-btn">⚙ Adjust settings &amp; re-run</button>
          <button class="btn-text" id="reset-btn">Start over with a different file</button>
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

    document.getElementById("adjust-btn").addEventListener("click", () => {
      if (lastLoaded) {
        presentColumnPicker(lastLoaded.path, lastLoaded.displayName, lastLoaded.byteSize, target || null);
        if (!settingsPanel.classList.contains("open")) {
          settingsToggle.classList.add("open");
          settingsPanel.classList.add("open");
        }
        stepsEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
    document.getElementById("reset-btn").addEventListener("click", () => {
      stepsEl.innerHTML = "";
      lastLoaded = null;
    });
    iframe.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    const progressEl = document.getElementById("analyze-progress");
    if (progressEl) progressEl.remove();
    renderStepError("Analysis crashed", err);
  }
}

/* ---------- File input / drag-drop wiring ---------- */

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

document.querySelectorAll(".example-card").forEach((card) => {
  card.addEventListener("click", () => {
    handleSampleClick(card.dataset.sample, card.dataset.target);
    const liveDemoSlide = document.getElementById("dropzone")?.closest(".slide");
    if (liveDemoSlide) goToSlide(Number(liveDemoSlide.dataset.slide));
  });
});

/* ---------- Slide / stepper navigation ---------- */

const SLIDE_LABELS = ["Overview", "How it works", "Example Sheets", "Live demo", "Why it's real"];
const slides = [...document.querySelectorAll(".slide")];
const dots = [...document.querySelectorAll(".slide-dot")];
const prevBtn = document.getElementById("prev-slide");
const nextBtn = document.getElementById("next-slide");
const slideLabel = document.getElementById("slide-label");
let currentSlide = 0;

function goToSlide(index) {
  index = Math.max(0, Math.min(slides.length - 1, index));
  currentSlide = index;
  slides.forEach((s) => s.classList.toggle("active", Number(s.dataset.slide) === index));
  dots.forEach((d) => d.classList.toggle("current", Number(d.dataset.slide) === index));
  document.querySelectorAll(".nav-links .slide-link").forEach((btn) => {
    btn.classList.toggle("current", Number(btn.dataset.slide) === index);
  });
  slideLabel.textContent = `${index + 1} / ${slides.length} — ${SLIDE_LABELS[index]}`;
  prevBtn.disabled = index === 0;
  nextBtn.disabled = index === slides.length - 1;
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
}

document.querySelectorAll(".slide-link, .slide-dot").forEach((el) => {
  el.addEventListener("click", () => goToSlide(Number(el.dataset.slide)));
});
prevBtn.addEventListener("click", () => goToSlide(currentSlide - 1));
nextBtn.addEventListener("click", () => goToSlide(currentSlide + 1));

document.addEventListener("keydown", (e) => {
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (e.key === "ArrowRight") goToSlide(currentSlide + 1);
  if (e.key === "ArrowLeft") goToSlide(currentSlide - 1);
});

goToSlide(0);
boot();
