(() => {
  const $ = (id) => document.getElementById(id);

  const els = {
    weightFile: $("weightFile"),
    classFile: $("classFile"),
    expFile: $("expFile"),
    device: $("device"),
    btnLoad: $("btnLoad"),
    loadMsg: $("loadMsg"),
    gpuHint: $("gpuHint"),
    imageFile: $("imageFile"),
    folderFiles: $("folderFiles"),
    dirPath: $("dirPath"),
    videoFile: $("videoFile"),
    videoPath: $("videoPath"),
    rtspUrl: $("rtspUrl"),
    btnOpenImage: $("btnOpenImage"),
    btnOpenFolder: $("btnOpenFolder"),
    btnOpenDirPath: $("btnOpenDirPath"),
    btnOpenVideo: $("btnOpenVideo"),
    btnOpenVideoPath: $("btnOpenVideoPath"),
    btnOpenRtsp: $("btnOpenRtsp"),
    btnPrevImage: $("btnPrevImage"),
    btnNextImage: $("btnNextImage"),
    galleryNav: $("galleryNav"),
    galleryInfo: $("galleryInfo"),
    recordToggle: $("recordToggle"),
    recordDuration: $("recordDuration"),
    recordStatus: $("recordStatus"),
    btnDownload: $("btnDownload"),
    sourceMsg: $("sourceMsg"),
    confSlider: $("confSlider"),
    nmsSlider: $("nmsSlider"),
    confVal: $("confVal"),
    nmsVal: $("nmsVal"),
    preview: $("preview"),
    stageEmpty: $("stageEmpty"),
    btnPlay: $("btnPlay"),
    btnPause: $("btnPause"),
    btnStop: $("btnStop"),
    statusPill: $("statusPill"),
    fpsMeta: $("fpsMeta"),
    detMeta: $("detMeta"),
    sessionMeta: $("sessionMeta"),
    frameInfo: $("frameInfo"),
    classChips: $("classChips"),
    timeline: $("timeline"),
    timeCurrent: $("timeCurrent"),
    timeDuration: $("timeDuration"),
    speedSelect: $("speedSelect"),
  };

  const SESSION_KEY = "nava_mv_session_id";
  let sessionId = localStorage.getItem(SESSION_KEY) || "";
  let pollTimer = null;
  let streamMode = false;
  let modelReady = false;
  let scrubbing = false;
  let lastSeekable = false;

  function rememberSession(id) {
    if (!id) return;
    sessionId = id;
    localStorage.setItem(SESSION_KEY, id);
    if (els.sessionMeta) {
      els.sessionMeta.textContent = `session ${id.slice(0, 8)}`;
    }
  }

  function withSessionQuery(url) {
    if (!sessionId) return url;
    const join = url.includes("?") ? "&" : "?";
    return `${url}${join}session_id=${encodeURIComponent(sessionId)}`;
  }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`pane-${tab.dataset.tab}`).classList.add("active");
    });
  });

  function setHint(el, text, kind) {
    el.textContent = text || "";
    el.classList.remove("ok", "error");
    if (kind) el.classList.add(kind);
  }

  function showFrame(url) {
    const stage = els.preview.parentElement;
    els.preview.src = url;
    els.preview.classList.add("visible");
    stage.classList.add("has-frame");
  }

  function startMjpeg() {
    streamMode = true;
    showFrame(withSessionQuery(`/api/stream?ts=${Date.now()}`));
  }

  function refreshStill() {
    showFrame(withSessionQuery(`/api/frame.jpg?ts=${Date.now()}`));
  }

  function formatTime(sec) {
    sec = Math.max(0, Number(sec) || 0);
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (sessionId) headers.set("X-Session-Id", sessionId);
    const res = await fetch(path, { ...options, headers });
    const data = await res.json().catch(() => ({}));
    if (data.session_id) rememberSession(data.session_id);
    if (!res.ok) {
      const detail = data.detail || res.statusText || "Request failed";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  async function ensureSession() {
    const data = await api("/api/session", { method: "POST" });
    rememberSession(data.session_id);
    if (els.sessionMeta && data.active_sessions != null) {
      els.sessionMeta.textContent = `session ${sessionId.slice(0, 8)} · ${data.active_sessions}/${data.max_sessions} users`;
    }
    return data;
  }

  async function uploadModel() {
    if (!els.weightFile.files[0] || !els.classFile.files[0] || !els.expFile.files[0]) {
      setHint(els.loadMsg, "Please select weight, class, and experiment files.", "error");
      return;
    }
    els.btnLoad.disabled = true;
    setHint(els.loadMsg, "Loading model…");
    try {
      const fd = new FormData();
      fd.append("weight", els.weightFile.files[0]);
      fd.append("classes", els.classFile.files[0]);
      fd.append("exp", els.expFile.files[0]);
      fd.append("device", els.device.value);
      fd.append("conf", els.confSlider.value);
      fd.append("nms", els.nmsSlider.value);

      const data = await api("/api/upload", { method: "POST", body: fd });
      modelReady = true;
      els.confSlider.value = data.conf;
      els.nmsSlider.value = data.nms;
      els.confVal.textContent = Number(data.conf).toFixed(2);
      els.nmsVal.textContent = Number(data.nms).toFixed(2);
      renderClasses(data.class_names || []);
      setHint(
        els.loadMsg,
        `Loaded (${data.backend}) on ${data.device}${data.fp16 ? " FP16" : ""} · ${data.num_classes} classes · input ${data.test_size}`,
        "ok"
      );
      pollStatus();
    } catch (err) {
      modelReady = false;
      setHint(els.loadMsg, err.message, "error");
    } finally {
      els.btnLoad.disabled = false;
    }
  }

  function renderClasses(names) {
    els.classChips.innerHTML = "";
    names.slice(0, 40).forEach((name) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = name;
      els.classChips.appendChild(chip);
    });
    if (names.length > 40) {
      const more = document.createElement("span");
      more.className = "chip";
      more.textContent = `+${names.length - 40} more`;
      els.classChips.appendChild(more);
    }
  }

  async function openFileSource(input, sourceType) {
    if (!modelReady) {
      setHint(els.sourceMsg, "Load the model first.", "error");
      return;
    }
    if (!input.files[0]) {
      setHint(els.sourceMsg, `Choose a ${sourceType} file.`, "error");
      return;
    }
    setHint(els.sourceMsg, `Opening ${sourceType}…`);
    try {
      const fd = new FormData();
      fd.append("file", input.files[0]);
      fd.append("source_type", sourceType);
      const data = await api("/api/source/file", { method: "POST", body: fd });
      setHint(els.sourceMsg, `${sourceType} ready.`, "ok");
      if (sourceType === "image") {
        streamMode = false;
        refreshStill();
      } else {
        startMjpeg();
      }
      // Apply current speed after opening video
      if (sourceType === "video") {
        await api("/api/control/speed", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ speed: Number(els.speedSelect.value) }),
        });
      }
      updateStatusUI(data);
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }

  async function openRtsp() {
    if (!modelReady) {
      setHint(els.sourceMsg, "Load the model first.", "error");
      return;
    }
    const url = els.rtspUrl.value.trim();
    if (!url) {
      setHint(els.sourceMsg, "Enter an RTSP / media URL.", "error");
      return;
    }
    setHint(els.sourceMsg, "Connecting…");
    try {
      const data = await api("/api/source/rtsp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, source_type: "rtsp" }),
      });
      setHint(els.sourceMsg, "Source connected.", "ok");
      startMjpeg();
      updateStatusUI(data);
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }

  async function openFolderUpload() {
    if (!modelReady) {
      setHint(els.sourceMsg, "Load the model first.", "error");
      return;
    }
    const files = els.folderFiles.files;
    if (!files || !files.length) {
      setHint(els.sourceMsg, "Choose an image folder.", "error");
      return;
    }
    setHint(els.sourceMsg, `Opening folder (${files.length} files)…`);
    try {
      const fd = new FormData();
      for (const file of files) {
        // Only send image files from the directory picker
        if (file.type.startsWith("image/") || /\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name)) {
          fd.append("files", file, file.name);
        }
      }
      if (![...fd.keys()].length) {
        setHint(els.sourceMsg, "No image files in the selected folder.", "error");
        return;
      }
      const data = await api("/api/source/folder", { method: "POST", body: fd });
      setHint(
        els.sourceMsg,
        `Folder ready · ${data.image_count || 0} images. Use Prev / Next.`,
        "ok"
      );
      streamMode = false;
      refreshStill();
      updateStatusUI(data);
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }

  async function openDirPath() {
    if (!modelReady) {
      setHint(els.sourceMsg, "Load the model first.", "error");
      return;
    }
    const path = els.dirPath.value.trim();
    if (!path) {
      setHint(els.sourceMsg, "Enter a server directory path.", "error");
      return;
    }
    setHint(els.sourceMsg, "Opening directory…");
    try {
      const data = await api("/api/source/directory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      setHint(
        els.sourceMsg,
        `Directory ready · ${data.image_count || 0} images. Use Prev / Next.`,
        "ok"
      );
      streamMode = false;
      refreshStill();
      updateStatusUI(data);
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }

  async function openVideoPath() {
    if (!modelReady) {
      setHint(els.sourceMsg, "Load the model first.", "error");
      return;
    }
    const path = (els.videoPath && els.videoPath.value.trim()) || "";
    if (!path) {
      setHint(els.sourceMsg, "Enter a server video path.", "error");
      return;
    }
    setHint(els.sourceMsg, "Opening video from disk (no copy)…");
    try {
      const data = await api("/api/source/path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, source_type: "video" }),
      });
      setHint(els.sourceMsg, "Video opened from server path (not saved to uploads).", "ok");
      startMjpeg();
      await api("/api/control/speed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed: Number(els.speedSelect.value) }),
      });
      updateStatusUI(data);
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }

  async function stepImage(direction) {
    const endpoint =
      direction === "next" ? "/api/control/next-image" : "/api/control/prev-image";
    try {
      const s = await api(endpoint, { method: "POST" });
      streamMode = false;
      refreshStill();
      updateStatusUI(s);
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }

  function updateGallery(s) {
    const gallery = !!s.image_gallery;
    els.galleryNav.hidden = !gallery;
    if (!gallery) return;

    const idx = (Number(s.image_index) || 0) + 1;
    const count = Number(s.image_count) || 0;
    const name = s.image_name || "image";
    els.galleryInfo.textContent = `${idx} / ${count} · ${name}`;
    els.btnPrevImage.disabled = !s.has_prev;
    els.btnNextImage.disabled = !s.has_next;
  }

  function updateRecordUI(s) {
    const recording = !!s.recording;
    const ready = !!s.download_ready;
    // Keep checkbox in sync unless user is mid-click (handled by change handler)
    if (els.recordToggle.checked !== recording && !els.recordToggle.dataset.busy) {
      els.recordToggle.checked = recording;
    }

    els.recordStatus.classList.remove("live", "ready");
    if (recording) {
      const elapsed = s.record_elapsed_sec ?? 0;
      const limit = s.record_limit_sec;
      els.recordStatus.textContent = limit
        ? `recording ${elapsed}s / ${limit}s`
        : `recording ${elapsed}s`;
      els.recordStatus.classList.add("live");
    } else if (ready) {
      els.recordStatus.textContent = `ready · ${s.download_name || "video"}`;
      els.recordStatus.classList.add("ready");
    } else {
      els.recordStatus.textContent = "off";
    }

    if (ready) {
      els.btnDownload.hidden = false;
      els.btnDownload.href = withSessionQuery(`/api/record/download?ts=${Date.now()}`);
      els.btnDownload.setAttribute("download", s.download_name || "inference.mp4");
    } else {
      els.btnDownload.hidden = true;
    }

    const canRecord =
      s.source_type === "video" || s.source_type === "rtsp";
    els.recordToggle.disabled = !canRecord && !recording;
    els.recordDuration.disabled = recording;
  }

  function updateTimeline(s) {
    const seekable = !!s.seekable;
    lastSeekable = seekable;
    els.timeline.disabled = !seekable;
    els.speedSelect.disabled =
      s.source_type === "image" || s.source_type === "image_dir";

    const total = Number(s.total_frames) || 0;
    const frame = Number(s.frame) || 0;
    const duration = Number(s.duration_sec) || 0;
    const current = Number(s.current_sec) || 0;

    if (seekable && total > 0) {
      els.timeline.max = String(Math.max(0, total - 1));
      if (!scrubbing) {
        els.timeline.value = String(Math.min(frame, total - 1));
      }
    } else {
      els.timeline.max = "0";
      if (!scrubbing) els.timeline.value = "0";
    }

    const shownCurrent = scrubbing
      ? (Number(els.timeline.value) / (Number(s.source_fps) || 25))
      : current;
    els.timeCurrent.textContent = formatTime(shownCurrent);
    els.timeDuration.textContent = formatTime(duration);

    if (s.speed != null && String(s.speed) !== els.speedSelect.value) {
      const opt = [...els.speedSelect.options].find(
        (o) => Math.abs(Number(o.value) - Number(s.speed)) < 1e-6
      );
      if (opt) els.speedSelect.value = opt.value;
    }
  }

  function updateStatusUI(s) {
    if (!s) return;
    if (s.session_id) rememberSession(s.session_id);
    const status = s.status || "idle";
    els.statusPill.textContent = status;
    els.statusPill.classList.remove("playing", "error");
    if (status === "playing") els.statusPill.classList.add("playing");
    if (status === "error") els.statusPill.classList.add("error");
    els.fpsMeta.textContent = `FPS ${s.fps ?? "—"}`;
    els.detMeta.textContent = `Dets ${s.detections ?? "—"}`;
    if (els.sessionMeta && s.sessions) {
      els.sessionMeta.textContent = `session ${(s.session_id || sessionId).slice(0, 8)} · ${s.sessions.active_sessions}/${s.sessions.max_sessions} users`;
    }

    const speedLabel = s.speed != null ? ` · ${s.speed}×` : "";
    if (s.image_gallery) {
      els.frameInfo.textContent = `image ${(s.image_index ?? 0) + 1} / ${s.image_count ?? 0}`;
    } else if (s.total_frames) {
      els.frameInfo.textContent = `frame ${s.frame ?? 0} / ${s.total_frames}${speedLabel}`;
    } else {
      els.frameInfo.textContent = `frame ${s.frame ?? "—"}${speedLabel}`;
    }

    updateTimeline(s);
    updateGallery(s);
    updateRecordUI(s);
    if (s.error) setHint(els.sourceMsg, s.error, "error");
  }

  async function pollStatus() {
    try {
      const s = await api("/api/status");
      updateStatusUI(s);
      if (s.model_loaded) modelReady = true;
      if (!streamMode && (s.source_type === "image" || s.source_type === "image_dir")) {
        // keep still image / gallery frame
      } else if (s.playing || s.status === "ready" || s.status === "paused") {
        if (
          !streamMode &&
          s.source_type &&
          s.source_type !== "image" &&
          s.source_type !== "image_dir"
        ) {
          startMjpeg();
        }
      }
    } catch (_) {
      /* ignore transient */
    }
  }

  function debounce(fn, ms) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  }

  const pushConf = debounce(async () => {
    const conf = Number(els.confSlider.value);
    try {
      await api("/api/conf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conf }),
      });
      if (!streamMode) refreshStill();
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }, 200);

  const pushNms = debounce(async () => {
    const nms = Number(els.nmsSlider.value);
    try {
      await api("/api/nms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nms }),
      });
      if (!streamMode) refreshStill();
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }, 200);

  const pushSeek = debounce(async (frame) => {
    try {
      const s = await api("/api/control/seek", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: Number(frame) }),
      });
      updateStatusUI(s);
      if (!s.playing) {
        streamMode = false;
        refreshStill();
      }
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    } finally {
      scrubbing = false;
    }
  }, 120);

  els.btnLoad.addEventListener("click", uploadModel);
  els.btnOpenImage.addEventListener("click", () => openFileSource(els.imageFile, "image"));
  els.btnOpenFolder.addEventListener("click", openFolderUpload);
  els.btnOpenDirPath.addEventListener("click", openDirPath);
  els.btnOpenVideo.addEventListener("click", () => openFileSource(els.videoFile, "video"));
  if (els.btnOpenVideoPath) {
    els.btnOpenVideoPath.addEventListener("click", openVideoPath);
  }
  els.btnOpenRtsp.addEventListener("click", openRtsp);
  els.btnPrevImage.addEventListener("click", () => stepImage("prev"));
  els.btnNextImage.addEventListener("click", () => stepImage("next"));

  els.recordToggle.addEventListener("change", async () => {
    els.recordToggle.dataset.busy = "1";
    try {
      if (els.recordToggle.checked) {
        const dur = Number(els.recordDuration.value);
        const body = {
          duration_sec: Number.isFinite(dur) && dur > 0 ? dur : 0,
        };
        const s = await api("/api/record/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        updateStatusUI(s);
        setHint(
          els.sourceMsg,
          body.duration_sec
            ? `Recording inference for ${body.duration_sec}s…`
            : "Recording until you turn it off…",
          "ok"
        );
      } else {
        const s = await api("/api/record/stop", { method: "POST" });
        updateStatusUI(s);
        if (s.download_ready) {
          setHint(els.sourceMsg, "Recording saved — click Download.", "ok");
        }
      }
    } catch (err) {
      els.recordToggle.checked = false;
      setHint(els.sourceMsg, err.message, "error");
    } finally {
      delete els.recordToggle.dataset.busy;
    }
  });

  document.addEventListener("keydown", (ev) => {
    if (!els.galleryNav || els.galleryNav.hidden) return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.key === "ArrowRight") {
      ev.preventDefault();
      stepImage("next");
    } else if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      stepImage("prev");
    }
  });

  els.btnPlay.addEventListener("click", async () => {
    try {
      const s = await api("/api/control/play", { method: "POST" });
      startMjpeg();
      updateStatusUI(s);
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  });

  els.btnPause.addEventListener("click", async () => {
    try {
      const s = await api("/api/control/pause", { method: "POST" });
      updateStatusUI(s);
      refreshStill();
      streamMode = false;
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  });

  els.btnStop.addEventListener("click", async () => {
    try {
      const s = await api("/api/control/stop", { method: "POST" });
      updateStatusUI(s);
      streamMode = false;
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  });

  els.confSlider.addEventListener("input", () => {
    els.confVal.textContent = Number(els.confSlider.value).toFixed(2);
    pushConf();
  });

  els.nmsSlider.addEventListener("input", () => {
    els.nmsVal.textContent = Number(els.nmsSlider.value).toFixed(2);
    pushNms();
  });

  els.speedSelect.addEventListener("change", async () => {
    try {
      const s = await api("/api/control/speed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed: Number(els.speedSelect.value) }),
      });
      updateStatusUI(s);
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  });

  els.timeline.addEventListener("pointerdown", () => {
    if (!els.timeline.disabled) scrubbing = true;
  });

  els.timeline.addEventListener("input", () => {
    if (els.timeline.disabled) return;
    scrubbing = true;
    const fps = 25; // preview label; refined on status poll
    const frame = Number(els.timeline.value);
    const max = Number(els.timeline.max) || 1;
    const durationText = els.timeDuration.textContent;
    // Estimate time from slider fraction of duration label if possible
    els.timeCurrent.textContent = formatTime(
      (frame / Math.max(max, 1)) * (parseTime(durationText) || frame / fps)
    );
  });

  els.timeline.addEventListener("change", () => {
    if (els.timeline.disabled) return;
    scrubbing = true;
    pushSeek(els.timeline.value);
  });

  els.timeline.addEventListener("pointerup", () => {
    if (scrubbing && !els.timeline.disabled) {
      pushSeek(els.timeline.value);
    }
  });

  function parseTime(label) {
    if (!label || !label.includes(":")) return 0;
    const [m, s] = label.split(":").map(Number);
    return (m || 0) * 60 + (s || 0);
  }

  async function checkGpu() {
    if (!els.gpuHint) return;
    try {
      const s = await api("/api/system");
      if (s.cuda_available) {
        const mem = s.gpu_memory_gb ? ` · ${s.gpu_memory_gb} GB` : "";
        const slots = ` · sessions ${s.active_sessions}/${s.max_sessions}`;
        els.gpuHint.textContent = `GPU ready: ${s.gpu_name}${mem}${slots}`;
        els.gpuHint.classList.add("ok");
        if (els.device.value === "auto" || els.device.value === "cpu") {
          els.device.value = "auto";
        }
      } else {
        els.gpuHint.textContent =
          "No GPU in this runtime — inference will use CPU (slow). Enable Docker GPU / select CUDA.";
        els.gpuHint.classList.add("error");
      }
    } catch (_) {
      els.gpuHint.textContent = "Could not detect GPU status.";
    }
  }

  // Init private session, then poll
  ensureSession()
    .then(() => {
      checkGpu();
      pollStatus();
      pollTimer = setInterval(pollStatus, 500);
    })
    .catch((err) => {
      if (els.gpuHint) {
        els.gpuHint.textContent = err.message;
        els.gpuHint.classList.add("error");
      }
      setHint(els.loadMsg, err.message, "error");
    });

  window.addEventListener("beforeunload", () => {
    if (!sessionId) return;
    // Best-effort release (may not always fire)
    try {
      navigator.sendBeacon?.(
        "/api/session",
        new Blob([], { type: "application/json" })
      );
    } catch (_) {
      /* ignore */
    }
  });
})();
