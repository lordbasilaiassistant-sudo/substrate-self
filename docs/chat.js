// Eli chat — runs the ONNX-exported Eli in the browser via onnxruntime-web.
// No backend. The 7.4 MB model is loaded once from this static page.
// Generation is autoregressive char sampling (the production model is char-level).

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const status = $("status");
  const meta = $("meta");
  const metaText = $("meta-text");
  const transcript = $("transcript");
  const form = $("input-form");
  const prompt = $("prompt");
  const send = $("send");
  const tempInput = $("temp");
  const tempVal = $("temp-val");
  const maxNewInput = $("max-new");
  const clearBtn = $("clear");
  const manifestPre = $("manifest-pre");

  tempInput.addEventListener("input", () => {
    tempVal.textContent = Number(tempInput.value).toFixed(2);
  });

  const SPECIAL_TOKENS = new Set(["<pad>", "<bos>", "<eos>", "<unk>"]);
  const NEWLINE_USER_PREFIX = "\nUser:";

  let session = null;
  let vocab = null;   // index -> char
  let stoi = null;    // char -> index
  let cfg = null;     // {block_size, vocab_size, ...}
  let busy = false;

  function setStatus(text, cls = "") {
    status.classList.remove("status-loading", "status-ready", "status-error");
    if (cls) status.classList.add(`status-${cls}`);
    status.textContent = text;
  }

  function showMeta(text) {
    meta.classList.remove("hidden");
    metaText.textContent = text;
  }

  function addTurn(role, text) {
    const el = document.createElement("div");
    el.className = `turn ${role}`;
    const label = document.createElement("span");
    label.className = "role";
    label.textContent = role === "user" ? "you" : "eli";
    el.appendChild(label);
    const body = document.createElement("span");
    body.className = "body";
    body.textContent = text;
    el.appendChild(body);
    transcript.appendChild(el);
    transcript.scrollIntoView({ behavior: "smooth", block: "end" });
    return body;
  }

  function encode(text) {
    const ids = [];
    const unk = stoi["<unk>"] ?? 0;
    for (const ch of text) ids.push(stoi[ch] ?? unk);
    return ids;
  }

  function decodeId(id) {
    const tok = vocab[id];
    if (!tok || SPECIAL_TOKENS.has(tok)) return "";
    return tok;
  }

  function softmaxTopK(logits, temperature, topK) {
    // logits: Float32Array of length vocab_size
    // returns sampled id
    const V = logits.length;
    const scaled = new Float32Array(V);
    for (let i = 0; i < V; i++) scaled[i] = logits[i] / Math.max(temperature, 1e-6);

    // top-k: copy, partial-sort. For small V (~69) this is trivial.
    const idx = new Int32Array(V);
    for (let i = 0; i < V; i++) idx[i] = i;
    idx.sort((a, b) => scaled[b] - scaled[a]);
    const k = Math.min(topK, V);
    const threshold = scaled[idx[k - 1]];

    let maxLogit = -Infinity;
    for (let i = 0; i < V; i++) {
      if (scaled[i] < threshold) scaled[i] = -Infinity;
      else if (scaled[i] > maxLogit) maxLogit = scaled[i];
    }

    let sum = 0;
    const probs = new Float32Array(V);
    for (let i = 0; i < V; i++) {
      probs[i] = scaled[i] === -Infinity ? 0 : Math.exp(scaled[i] - maxLogit);
      sum += probs[i];
    }
    if (sum === 0) {
      // fallback: argmax (shouldn't happen)
      let best = 0, bestV = -Infinity;
      for (let i = 0; i < V; i++) if (logits[i] > bestV) { bestV = logits[i]; best = i; }
      return best;
    }
    let r = Math.random() * sum;
    for (let i = 0; i < V; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return V - 1;
  }

  async function generate(promptText, maxNewTokens, temperature, topK, onChunk) {
    if (!session) return "";
    const blockSize = cfg.block_size;

    // Build the "User: ...\nEli:" prompt — same shape the model was trained on.
    const wrapped = `User: ${promptText}\nEli:`;
    let ids = encode(wrapped);

    let outText = "";
    const eosId = stoi["<eos>"];

    for (let step = 0; step < maxNewTokens; step++) {
      const ctx = ids.length > blockSize ? ids.slice(ids.length - blockSize) : ids;
      const T = ctx.length;
      const input = new ort.Tensor(
        "int64",
        BigInt64Array.from(ctx, (v) => BigInt(v)),
        [1, T]
      );
      let result;
      try {
        result = await session.run({ input_ids: input });
      } catch (e) {
        console.error("inference error", e);
        break;
      }
      const logitsTensor = result.logits; // shape [1, T, vocab]
      const V = cfg.vocab_size;
      // last-position logits = data[(T-1) * V .. T*V]
      const data = logitsTensor.data; // Float32Array
      const lastStart = (T - 1) * V;
      const slice = data.subarray(lastStart, lastStart + V);

      const nextId = softmaxTopK(slice, temperature, topK);
      if (nextId === eosId) break;
      ids.push(nextId);
      const piece = decodeId(nextId);
      if (piece) {
        outText += piece;
        if (onChunk) onChunk(piece, outText);
        // Stop at the next User: turn boundary
        if (outText.endsWith(NEWLINE_USER_PREFIX)) {
          outText = outText.slice(0, -NEWLINE_USER_PREFIX.length);
          break;
        }
      }
      // Yield to UI every few steps.
      if (step % 4 === 3) await new Promise((r) => setTimeout(r, 0));
    }
    return outText.trim();
  }

  async function loadManifest() {
    try {
      const r = await fetch("eli_manifest.json", { cache: "no-cache" });
      if (!r.ok) throw new Error("manifest fetch failed");
      const m = await r.json();
      manifestPre.textContent = JSON.stringify(m, null, 2);
      return m;
    } catch (e) {
      manifestPre.textContent = "(manifest not available)";
      return null;
    }
  }

  async function loadTokenizer() {
    const r = await fetch("tokenizer.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("tokenizer fetch failed");
    const data = await r.json();
    vocab = data.vocab;
    stoi = {};
    for (let i = 0; i < vocab.length; i++) stoi[vocab[i]] = i;
    return vocab.length;
  }

  async function loadSession() {
    // Prefer WebGPU, fall back to WASM.
    const providers = ["webgpu", "wasm"];
    try {
      session = await ort.InferenceSession.create("eli.onnx", {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      return session;
    } catch (e) {
      console.warn("session create failed, retrying wasm only", e);
      session = await ort.InferenceSession.create("eli.onnx", {
        executionProviders: ["wasm"],
      });
      return session;
    }
  }

  async function boot() {
    setStatus("Loading manifest…", "loading");
    const manifest = await loadManifest();
    if (manifest) {
      cfg = {
        block_size: manifest.block_size,
        vocab_size: manifest.vocab_size,
      };
    }
    setStatus("Loading tokenizer…", "loading");
    const vsize = await loadTokenizer();
    if (!cfg) cfg = { block_size: 128, vocab_size: vsize };

    setStatus("Loading Eli (one-time 7.4 MB download)…", "loading");
    try {
      await loadSession();
    } catch (e) {
      console.error(e);
      setStatus("Failed to load Eli model. " + e.message, "error");
      return;
    }

    const provider = session.handler?.backendName || "wasm";
    setStatus("Eli is ready.", "ready");
    showMeta(
      `vocab=${cfg.vocab_size} · ctx=${cfg.block_size} · ` +
      `backend=${provider} · model sha256=${(manifest?.onnx_sha256 ?? "").slice(0,12)}…`
    );

    prompt.disabled = false;
    send.disabled = false;
    prompt.focus();
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (busy || !session) return;
    const text = prompt.value.trim();
    if (!text) return;
    busy = true;
    prompt.value = "";
    send.disabled = true;
    prompt.disabled = true;

    addTurn("user", text);
    const eliBody = addTurn("eli", "");
    eliBody.textContent = "";
    const cursor = document.createElement("span");
    cursor.textContent = "▍";
    cursor.style.opacity = "0.5";
    eliBody.appendChild(cursor);

    const temperature = Number(tempInput.value);
    const maxNew = Math.min(400, Math.max(20, Number(maxNewInput.value) || 120));
    const topK = 40;

    let buf = "";
    try {
      await generate(text, maxNew, temperature, topK, (piece, full) => {
        buf = full;
        // Insert before the cursor.
        eliBody.firstChild &&
          eliBody.insertBefore(document.createTextNode(piece), cursor);
      });
    } catch (e) {
      console.error(e);
      buf = "(generation error: " + e.message + ")";
      eliBody.textContent = buf;
    } finally {
      cursor.remove();
      if (!buf || buf.length === 0) {
        eliBody.textContent = "(no output — try increasing tokens or temperature)";
      }
      busy = false;
      prompt.disabled = false;
      send.disabled = false;
      prompt.focus();
    }
  });

  clearBtn.addEventListener("click", () => {
    transcript.innerHTML = "";
    prompt.focus();
  });

  // Configure ONNX Runtime Web to fetch wasm from the same CDN as ort.min.js.
  if (window.ort && ort.env && ort.env.wasm) {
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  }

  boot().catch((e) => {
    console.error(e);
    setStatus("Boot error: " + e.message, "error");
  });
})();
