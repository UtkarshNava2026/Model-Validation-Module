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
    modeSingle: $("modeSingle"),
    modeLive: $("modeLive"),
    singleSource: $("singleSource"),
    liveSource: $("liveSource"),
    liveGrid: $("liveGrid"),
    liveEmpty: $("liveEmpty"),
    liveMsg: $("liveMsg"),
    btnLiveStart: $("btnLiveStart"),
    btnLiveStop: $("btnLiveStop"),
    playbackBuilder: $("playbackBuilder"),
    pbApply: $("pbApply"),
    pbMsg: $("pbMsg"),
    btnSettings: $("btnSettings"),
    settingsScrim: $("settingsScrim"),
  };

  const SESSION_KEY = "nava_mv_session_id";
  let sessionId = localStorage.getItem(SESSION_KEY) || "";
  let pollTimer = null;
  let streamMode = false;
  let modelReady = false;
  let scrubbing = false;
  let lastSeekable = false;
  let viewMode = "single";
  let modeChosen = false;
  let liveTilesConnected = 0;
  let settingsOpen = false;

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
    const res = await fetch(path, {
      ...options,
      headers,
      credentials: "same-origin",
    });
    const data = await res.json().catch(() => ({}));
    if (data.session_id) rememberSession(data.session_id);
    if (!res.ok) {
      const detail = data.detail || res.statusText || "Request failed";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  async function ensureSession() {
    const previous = sessionId;
    const data = await api("/api/session", { method: "POST" });
    rememberSession(data.session_id);
    if (els.sessionMeta && data.active_sessions != null) {
      const tag = data.resumed || (previous && previous === data.session_id) ? "resumed" : "new";
      els.sessionMeta.textContent =
        `session ${sessionId.slice(0, 8)} (${tag}) · ${data.active_sessions}/${data.max_sessions} users`;
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
      if (viewMode === "live") {
        clearLiveTiles();
        setHint(els.liveMsg, "Model reloaded. Start feeds again.", "ok");
      }
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

  function applyChrome(mode) {
    viewMode = mode === "live" ? "live" : "single";
    const live = viewMode === "live";
    document.querySelector(".app").classList.toggle("mode-live", live);
    if (els.singleSource) els.singleSource.hidden = live;
    if (els.liveSource) els.liveSource.hidden = !live;
    if (els.liveGrid) els.liveGrid.hidden = !live;
    if (els.modeSingle) {
      els.modeSingle.classList.toggle("active", !live);
      els.modeSingle.setAttribute("aria-pressed", String(!live));
    }
    if (els.modeLive) {
      els.modeLive.classList.toggle("active", live);
      els.modeLive.setAttribute("aria-pressed", String(live));
    }
    const heading = document.getElementById("sourceHeading");
    if (heading) heading.textContent = live ? "2 · Live feeds" : "2 · Input source";
    if (!live) setSettingsOpen(false);
  }

  function setSettingsOpen(open) {
    settingsOpen = viewMode === "live" && !!open;
    document.querySelector(".app").classList.toggle("settings-open", settingsOpen);
    if (els.btnSettings) {
      els.btnSettings.classList.toggle("active", settingsOpen);
      els.btnSettings.setAttribute("aria-expanded", String(settingsOpen));
      els.btnSettings.textContent = settingsOpen ? "Close" : "Settings";
    }
    if (els.settingsScrim) els.settingsScrim.hidden = !settingsOpen;
  }

  function clearLiveTiles() {
    liveTilesConnected = 0;
    if (!els.liveGrid) return;
    els.liveGrid.classList.remove("cols-1", "cols-2", "cols-3", "cols-4");
    els.liveGrid.classList.add("idle");
    if (els.liveEmpty) els.liveEmpty.hidden = false;
    els.liveGrid.querySelectorAll(".live-tile").forEach((tile) => {
      tile.hidden = true;
      tile.classList.remove("has-frame", "is-error");
      const img = tile.querySelector("img");
      if (img) {
        img.onerror = null;
        img.removeAttribute("src");
      }
      const err = tile.querySelector(".live-tile-error");
      if (err) {
        err.hidden = true;
        err.textContent = "";
      }
      const wait = tile.querySelector(".live-tile-wait");
      if (wait) wait.hidden = false;
    });
  }

  function attachLiveImage(img, slot) {
    let attempt = 0;
    img.onerror = () => {
      if (viewMode !== "live" || liveTilesConnected <= slot) return;
      attempt += 1;
      const wait = Math.min(8000, 700 * attempt);
      window.setTimeout(() => {
        if (viewMode !== "live" || liveTilesConnected <= slot) return;
        img.src = withSessionQuery(`/api/live/stream/${slot}?ts=${Date.now()}`);
      }, wait);
    };
    img.src = withSessionQuery(`/api/live/stream/${slot}?ts=${Date.now()}`);
  }

  function connectLiveTiles(count) {
    const n = Math.max(0, Math.min(4, Number(count) || 0));
    if (!n || !els.liveGrid || liveTilesConnected === n) return;
    liveTilesConnected = n;
    els.liveGrid.classList.remove("idle", "cols-1", "cols-2", "cols-3", "cols-4");
    els.liveGrid.classList.add(`cols-${n}`);
    if (els.liveEmpty) els.liveEmpty.hidden = true;
    els.liveGrid.querySelectorAll(".live-tile").forEach((tile, i) => {
      const img = tile.querySelector("img");
      tile.hidden = i >= n;
      if (i < n && img) {
        attachLiveImage(img, i);
      } else if (img) {
        img.onerror = null;
        img.removeAttribute("src");
      }
    });
  }

  function updateLiveHUD(live) {
    if (!live) return;
    const slots = live.slots || [];
    const running = !!live.running;
    const allDown = slots.length > 0 && slots.every((slot) => slot.error && !slot.has_frame);
    els.statusPill.textContent = running ? "live" : (modelReady ? "ready" : (live.status || "idle"));
    els.statusPill.classList.remove("playing", "error");
    if (running && !allDown) els.statusPill.classList.add("playing");
    if (allDown) els.statusPill.classList.add("error");

    const fpsVals = slots.map((slot) => Number(slot.fps)).filter((n) => n > 0);
    const dets = slots.reduce((sum, slot) => sum + (Number(slot.detections) || 0), 0);
    els.fpsMeta.textContent = fpsVals.length
      ? `FPS ${(fpsVals.reduce((a, b) => a + b, 0) / fpsVals.length).toFixed(1)}`
      : "FPS —";
    els.detMeta.textContent = running || slots.some((slot) => slot.has_frame)
      ? `Dets ${dets}`
      : "Dets —";

    if (running && slots.length) connectLiveTiles(slots.length);
    else if (!running && liveTilesConnected) clearLiveTiles();

    slots.forEach((slot) => {
      const tile = els.liveGrid.querySelector(`.live-tile[data-slot="${slot.index}"]`);
      if (!tile) return;
      const label = tile.querySelector(".live-label");
      const stats = tile.querySelector(".live-stats");
      const err = tile.querySelector(".live-tile-error");
      const wait = tile.querySelector(".live-tile-wait");
      if (label) label.textContent = slot.label || `Cam ${slot.index + 1}`;
      if (stats) {
        const fps = slot.fps ? Number(slot.fps).toFixed(1) : "—";
        stats.textContent = `FPS ${fps} · Dets ${slot.detections ?? "—"}`;
      }
      tile.classList.toggle("has-frame", !!slot.has_frame);
      tile.classList.toggle("is-error", !!slot.error);
      if (err) {
        err.hidden = !slot.error;
        err.textContent = slot.error || "";
      }
      if (wait) wait.hidden = !!slot.has_frame || !!slot.error;
    });
  }

  async function setMode(mode) {
    if (mode === viewMode) return;
    modeChosen = true;
    if (mode === "live") {
      streamMode = false;
      applyChrome("live");
      try {
        await api("/api/control/stop", { method: "POST" });
      } catch (_) {
        /* single source may already be idle */
      }
      return;
    }
    clearLiveTiles();
    applyChrome("single");
    try {
      await api("/api/live/stop", { method: "POST" });
    } catch (err) {
      setHint(els.sourceMsg, err.message, "error");
    }
  }

  let playbackTargetId = "";

  function hikStamp(value) {
    const match = String(value || "").match(
      /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/
    );
    if (!match) return "";
    const seconds = match[6] || "00";
    return `${match[1]}${match[2]}${match[3]}T${match[4]}${match[5]}${seconds}Z`;
  }

  function buildPlaybackUrl() {
    const ip = ($("pbIp").value || "").trim();
    const user = ($("pbUser").value || "").trim();
    const password = $("pbPass").value || "";
    const channel = ($("pbChannel").value || "").trim();
    const start = hikStamp($("pbStart").value);
    const end = hikStamp($("pbEnd").value);
    if (!ip || !user || !password || !channel || !start || !end) {
      throw new Error("Enter IP, user, password, channel, start, and end.");
    }
    if (!/^\d+$/.test(channel)) {
      throw new Error("Channel should be a number, for example 101.");
    }
    if (end <= start) {
      throw new Error("End must be after the start time.");
    }
    let host = ip.replace(/^rtsp:\/\//i, "").split("/")[0].trim();
    if (!host || /[\s@/?#]/.test(host)) {
      throw new Error("Enter the camera IP address only, for example 192.168.1.10.");
    }
    const authority = host.includes(":") ? host : `${host}:554`;
    return (
      `rtsp://${encodeURIComponent(user)}:${encodeURIComponent(password)}` +
      `@${authority}/Streaming/tracks/${channel}?starttime=${start}&endtime=${end}`
    );
  }

  function openPlaybackBuilder(inputId, button) {
    const input = document.getElementById(inputId);
    const builder = els.playbackBuilder;
    if (!input || !builder) return;
    if (playbackTargetId === inputId && !builder.hidden) {
      builder.hidden = true;
      playbackTargetId = "";
      document.querySelectorAll("[data-playback-for]").forEach((el) => {
        el.classList.remove("active");
      });
      return;
    }
    playbackTargetId = inputId;
    input.closest(".field").insertAdjacentElement("afterend", builder);
    builder.hidden = false;
    document.querySelectorAll("[data-playback-for]").forEach((el) => {
      el.classList.toggle("active", el === button);
    });
    setHint(els.pbMsg, "");
    $("pbIp").focus();
  }

  function applyPlaybackUrl() {
    const input = document.getElementById(playbackTargetId);
    if (!input) {
      setHint(els.pbMsg, "Choose a camera first.", "error");
      return;
    }
    try {
      input.value = buildPlaybackUrl();
      setHint(els.pbMsg, "Playback URL placed in this camera field.", "ok");
    } catch (err) {
      setHint(els.pbMsg, err.message, "error");
    }
  }

  function liveUrls() {
    return [0, 1, 2, 3]
      .map((i) => {
        const input = document.getElementById(`liveUrl${i}`);
        return input ? input.value.trim() : "";
      })
      .filter(Boolean);
  }

  async function startLive() {
    if (!modelReady) {
      setHint(els.liveMsg, "Load the model first.", "error");
      return;
    }
    const urls = liveUrls();
    if (!urls.length) {
      setHint(els.liveMsg, "Enter at least one RTSP URL.", "error");
      return;
    }
    els.btnLiveStart.disabled = true;
    setHint(els.liveMsg, "Connecting cameras…");
    try {
      const data = await api("/api/live/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls }),
      });
      const n = data.slot_count || urls.length;
      connectLiveTiles(n);
      updateLiveHUD(data);
      setHint(
        els.liveMsg,
        n === 1 ? "Live on 1 camera." : `Live on ${n} cameras.`,
        "ok"
      );
    } catch (err) {
      setHint(els.liveMsg, err.message, "error");
    } finally {
      els.btnLiveStart.disabled = false;
    }
  }

  async function stopLive() {
    try {
      const data = await api("/api/live/stop", { method: "POST" });
      clearLiveTiles();
      updateLiveHUD(data);
      setHint(els.liveMsg, "Feeds stopped.", "ok");
    } catch (err) {
      setHint(els.liveMsg, err.message, "error");
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
      if (s.model_loaded) modelReady = true;
      if (!modeChosen && s.live && s.live.running && viewMode !== "live") {
        applyChrome("live");
      }
      if (viewMode === "live") {
        if (s.session_id) rememberSession(s.session_id);
        if (els.sessionMeta && s.sessions) {
          els.sessionMeta.textContent =
            `session ${(s.session_id || sessionId).slice(0, 8)} · ${s.sessions.active_sessions}/${s.sessions.max_sessions} users`;
        }
        updateLiveHUD(s.live);
        return;
      }
      updateStatusUI(s);
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
      if (viewMode !== "live" && !streamMode) refreshStill();
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
      if (viewMode !== "live" && !streamMode) refreshStill();
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
  if (els.modeSingle) els.modeSingle.addEventListener("click", () => setMode("single"));
  if (els.modeLive) els.modeLive.addEventListener("click", () => setMode("live"));
  document.querySelectorAll("[data-playback-for]").forEach((button) => {
    button.addEventListener("click", () => {
      openPlaybackBuilder(button.dataset.playbackFor, button);
    });
  });
  if (els.pbApply) els.pbApply.addEventListener("click", applyPlaybackUrl);
  if (els.btnLiveStart) els.btnLiveStart.addEventListener("click", startLive);
  if (els.btnLiveStop) els.btnLiveStop.addEventListener("click", stopLive);
  if (els.btnSettings) {
    els.btnSettings.addEventListener("click", () => setSettingsOpen(!settingsOpen));
  }
  if (els.settingsScrim) {
    els.settingsScrim.addEventListener("click", () => setSettingsOpen(false));
  }
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
