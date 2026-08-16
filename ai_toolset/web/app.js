const $ = (sel) => document.querySelector(sel);

function el(tag, attrs, text) {
  const n = document.createElement(tag);
  if (attrs) Object.assign(n, attrs);
  if (text !== undefined) n.textContent = text;
  return n;
}

async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function setBadge(text, cls) {
  const b = $("#gpu-badge");
  b.textContent = text;
  b.className = "badge" + (cls ? " " + cls : "");
}

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#tab-" + t.dataset.tab).classList.add("active");
  });
});

async function loadHealth() {
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    const chips = $("#extras");
    chips.textContent = "";
    for (const [name, ok] of Object.entries(h.extras)) {
      chips.appendChild(el("span", { className: "chip " + (ok ? "yes" : "no") }, name));
    }
    const tbody = $("#gpu-table tbody");
    tbody.textContent = "";
    h.gpus.forEach((g) => {
      const tr = el("tr");
      tr.append(el("td", null, String(g.index)), el("td", null, g.name),
        el("td", null, g.driver), el("td", null, g.vram));
      tbody.appendChild(tr);
    });
    const gpuSel = $("#bench-gpus");
    gpuSel.textContent = "";
    (h.gpus.length ? h.gpus : [{ index: "cpu", name: "CPU" }]).forEach((g) => {
      const o = el("option", { value: g.index }, `GPU ${g.index} ${g.name}`);
      gpuSel.appendChild(o);
    });
    setBadge(h.gpus.length ? `GPU${h.gpus.length > 1 ? "s" : ""}: ` + h.gpus.map((g) => g.name).join(", ") : "CPU only", h.gpus.length ? "ok" : "warn");
  } catch (e) {
    setBadge("health check failed", "warn");
  }
}

$("#ocr-run").addEventListener("click", async () => {
  const out = $("#ocr-out");
  out.textContent = "running...";
  try {
    const src = document.querySelector('input[name="ocr-src"]:checked').value;
    const fd = new FormData();
    if (src === "file") {
      const f = $("#ocr-file").files[0];
      if (!f) throw new Error("choose an image first");
      fd.append("image", f);
    } else {
      fd.append("screen", "true");
    }
    const r = await fetch("/api/ocr", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    out.textContent = j.text || "(no text found)";
  } catch (e) {
    out.textContent = "ERROR: " + e.message;
  }
});

$("#det-run").addEventListener("click", async () => {
  const img = $("#det-img");
  const tbody = $("#det-table tbody");
  img.hidden = true;
  tbody.textContent = "";
  try {
    const f = $("#det-file").files[0];
    if (!f) throw new Error("choose an image first");
    const fd = new FormData();
    fd.append("image", f);
    fd.append("conf", $("#det-conf").value);
    const r = await fetch("/api/detect", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    img.src = "data:image/jpeg;base64," + j.annotated_b64;
    img.hidden = false;
    j.detections.forEach((d) => {
      const tr = el("tr");
      tr.append(el("td", null, d.label), el("td", null, d.conf.toFixed(2)),
        el("td", null, `[${d.box.join(",")}]`));
      tbody.appendChild(tr);
    });
    if (!j.detections.length) {
      const tr = el("tr");
      tr.append(el("td", { colSpan: 3 }, "no detections"));
      tbody.appendChild(tr);
    }
  } catch (e) {
    const tr = el("tr");
    tr.append(el("td", { colSpan: 3 }, "ERROR: " + e.message));
    tbody.appendChild(tr);
  }
});

$("#stt-run").addEventListener("click", async () => {
  const out = $("#stt-out");
  out.textContent = "transcribing...";
  try {
    const f = $("#stt-file").files[0];
    if (!f) throw new Error("choose an audio file first");
    const fd = new FormData();
    fd.append("audio", f);
    const r = await fetch("/api/transcribe", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    out.textContent = `[${j.language} p=${j.language_probability.toFixed(2)}]\n` + j.text;
  } catch (e) {
    out.textContent = "ERROR: " + e.message;
  }
});

$("#tts-run").addEventListener("click", async () => {
  const audio = $("#tts-audio");
  audio.hidden = true;
  try {
    const text = $("#tts-text").value.trim();
    if (!text) throw new Error("enter some text first");
    const fd = new FormData();
    fd.append("text", text);
    const spk = $("#tts-speaker").files[0];
    if (spk) fd.append("speaker", spk);
    const r = await fetch("/api/tts", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const blob = await r.blob();
    audio.src = URL.createObjectURL(blob);
    audio.hidden = false;
    audio.play();
  } catch (e) {
    alert("TTS error: " + e.message);
  }
});

$("#bench-run").addEventListener("click", async () => {
  const tbody = $("#bench-table tbody");
  tbody.textContent = "";
  try {
    const fd = new FormData();
    const sel = [...$("#bench-gpus").selectedOptions].map((o) => o.value);
    if (sel.length) fd.append("gpus_sel", sel.join(","));
    fd.append("iterations", $("#bench-iter").value);
    const img = $("#bench-img").files[0];
    if (img) fd.append("image", img);
    const r = await fetch("/api/benchmark", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    j.rows.forEach((row) => {
      const tr = el("tr");
      const isSTT = "mean_s" in row;
      tr.append(
        el("td", null, "gpu " + row.gpus),
        el("td", null, isSTT ? "stt" : "yolo"),
        el("td", null, row.model),
        el("td", null, (isSTT ? row.mean_s : row.mean_ms / 1000).toFixed(3) + "s"),
        el("td", null, (isSTT ? row.min_s : row.min_ms / 1000).toFixed(3) + "s"),
      );
      tbody.appendChild(tr);
    });
  } catch (e) {
    const tr = el("tr");
    tr.append(el("td", { colSpan: 5 }, "ERROR: " + e.message));
    tbody.appendChild(tr);
  }
});

let camRunning = false;
$("#cam-toggle").addEventListener("click", () => {
  const img = $("#cam-img");
  const btn = $("#cam-toggle");
  if (camRunning) {
    camRunning = false;
    img.hidden = true;
    img.removeAttribute("src");
    btn.textContent = "Start";
    return;
  }
  const sol = $("#cam-solution").value;
  img.src = "/api/webcam.mjpg?solution=" + sol + "&mirror=1&camera=0";
  img.hidden = false;
  camRunning = true;
  btn.textContent = "Stop";
});

$("#cam-solution").addEventListener("change", () => {
  if (camRunning) $("#cam-toggle").click();
});

loadHealth();
