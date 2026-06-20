// =====================================================================
// PSOBB Modding Suite - Audio perspective (2026-06-20)
//
// Replaces the old stub audio perspective (a bare /api/raw <audio>) with
// the audio suite:
//   * a codec badge (container + codec + ffmpeg availability)
//   * a .pac record-picker that re-points the <audio> element to
//     /api/audio/decode?record=i and redraws a waveform canvas from
//     /api/audio/waveform
//   * a DEV-only Replace file input (.wav / .ogg), shown ONLY when the
//     backend reports replace_supported (it writes DEV_DATA_DIR only).
//
// Routing: asset_router.js sends audio leaves here. The panel registers
// itself with the shared perspective registry after perspectives.js loads.
// =====================================================================
(function () {
  "use strict";

  if (!window.PSOPerspectives || !window.PSOPerspectives.register) {
    console.warn("[audio_panel] PSOPerspectives not available; audio perspective disabled");
    return;
  }
  var esc = window.PSOPerspectives.escapeHtml || function (s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  };

  function rawUrl(path) {
    return "/api/raw/" + String(path).split("/").map(encodeURIComponent).join("/");
  }
  // Last path segment (/ or \) — keeps absolute dev paths out of toasts.
  function basename(p) {
    if (p == null) return "";
    var s = String(p).replace(/[\\/]+$/, "");
    var i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
    return i >= 0 ? s.slice(i + 1) : s;
  }
  function apiUrl(base, path, extra) {
    var u = base + "?path=" + encodeURIComponent(path);
    if (extra) u += extra;
    return u;
  }
  function fmtDur(s) {
    if (!s || s <= 0) return "";
    return s < 1 ? (Math.round(s * 1000) + " ms") : (s.toFixed(2) + " s");
  }
  function fmtBytes(n) {
    if (typeof n !== "number" || n < 0) return "";
    if (n < 1024) return n + " B";
    var u = ["KB", "MB", "GB"], v = n / 1024, i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
    return v.toFixed(v < 10 ? 1 : 0) + " " + u[i];
  }

  // ---- waveform canvas paint ----------------------------------------
  function paintWaveform(canvas, wf) {
    if (!canvas || !wf) return;
    var ctx = canvas.getContext("2d");
    var W = canvas.width, H = canvas.height, mid = H / 2;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(120,140,160,0.4)";
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
    var n = (wf.min && wf.min.length) || 0;
    if (!n) {
      ctx.fillStyle = "rgba(200,200,200,0.5)";
      ctx.fillText("(no samples)", 8, mid);
      return;
    }
    var bw = W / n;
    // peaks (min..max) in a soft fill, rms overlaid darker.
    for (var i = 0; i < n; i++) {
      var x = i * bw;
      var yMax = mid - (wf.max[i] || 0) * mid;
      var yMin = mid - (wf.min[i] || 0) * mid;
      ctx.fillStyle = "rgba(90,170,255,0.55)";
      ctx.fillRect(x, yMax, Math.max(1, bw), Math.max(1, yMin - yMax));
      var r = (wf.rms[i] || 0) * mid;
      ctx.fillStyle = "rgba(40,110,200,0.9)";
      ctx.fillRect(x, mid - r, Math.max(1, bw), Math.max(1, r * 2));
    }
  }

  function loadWaveform(state, recordIdx) {
    var canvas = state.stage.querySelector("#audCanvas");
    if (!canvas) return;
    var url = apiUrl("/api/audio/waveform", state.path,
      "&record=" + recordIdx + "&buckets=" + Math.max(64, Math.floor(canvas.width)));
    fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (wf) {
      paintWaveform(canvas, wf);
    }).catch(function (e) {
      var ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "rgba(255,120,120,0.8)";
      ctx.fillText("waveform unavailable: " + (e && e.message || e), 8, canvas.height / 2);
    });
  }

  // ---- audio source selection ----------------------------------------
  function pointAudio(state, info, recordIdx) {
    var a = state.stage.querySelector("#audPlayer");
    if (!a) return;
    var kind = info.decode_kind;
    // .pac and .sfd MUST go through /api/audio/decode (no browser codec).
    // .ogg plays natively from /api/raw (browser Vorbis); .wav too.
    var src;
    if (kind === "pac") {
      src = apiUrl("/api/audio/decode", state.path, "&record=" + recordIdx);
    } else if (kind === "ogg" || kind === "wav") {
      src = rawUrl(state.path);
    } else { // sfd -> server-side decode (ffmpeg)
      src = apiUrl("/api/audio/decode", state.path, "&record=" + recordIdx);
    }
    a.src = src;
    a.load();
    a.play().catch(function () { /* autoplay block is fine */ });
    loadWaveform(state, recordIdx);
  }

  // ---- Replace (DEV-only) UI -----------------------------------------
  function wireReplace(state, info) {
    var form = state.insp.querySelector("#audReplaceForm");
    if (!form) return;
    var statusEl = state.insp.querySelector("#audReplaceStatus");
    function setStatus(msg, cls) {
      if (!statusEl) return;
      statusEl.className = "aud-replace-status " + (cls || "");
      statusEl.textContent = msg;
    }
    async function doReplace(deploy) {
      var fileInput = form.querySelector("#audReplaceFile");
      if (!fileInput || !fileInput.files || !fileInput.files.length) {
        setStatus("pick a .wav or .ogg file first", "err");
        return;
      }
      var fd = new FormData();
      fd.append("path", state.path);
      fd.append("record", String(state.currentRecord || 0));
      fd.append("deploy", deploy ? "true" : "false");
      if (form.querySelector("#audNormalize").checked) fd.append("normalize", "true");
      fd.append("file", fileInput.files[0]);
      setStatus(deploy ? "deploying to DEV…" : "building preview…", "");
      try {
        var r = await fetch("/api/audio/replace", { method: "POST", body: fd });
        var j = await r.json().catch(function () { return {}; });
        if (!r.ok) {
          var detail = (j && j.detail) || ("HTTP " + r.status);
          if (r.status === 501) detail = "ffmpeg required for this conversion: " + detail;
          setStatus("replace failed: " + detail, "err");
          return;
        }
        if (j.deployed) {
          setStatus("deployed to DEV: " + (basename(j.path) || "ok") +
            (j.backup_path ? " (backup saved)" : ""), "ok");
        } else {
          setStatus("preview ready", "ok");
          // Audition the preview by pointing the player at the export URL.
          var a = state.stage.querySelector("#audPlayer");
          if (a && j.export_url) { a.src = j.export_url; a.load(); a.play().catch(function () {}); }
          var dl = state.insp.querySelector("#audPreviewLink");
          if (dl && j.export_url) {
            dl.href = j.export_url; dl.style.display = "inline";
            dl.textContent = "download preview";
          }
        }
      } catch (e) {
        setStatus("replace error: " + (e && e.message || e), "err");
      }
    }
    var prevBtn = form.querySelector("#audReplacePreview");
    var depBtn = form.querySelector("#audReplaceDeploy");
    if (prevBtn) prevBtn.addEventListener("click", function (e) { e.preventDefault(); doReplace(false); });
    if (depBtn) depBtn.addEventListener("click", function (e) { e.preventDefault(); doReplace(true); });
  }

  function renderInspector(state, info) {
    var ff = info.ffmpeg ? '<span class="aud-badge ok">ffmpeg ✓</span>'
                         : '<span class="aud-badge warn">ffmpeg ✗</span>';
    var badge =
      '<div class="aud-badges">' +
      '<span class="aud-badge">' + esc(info.container) + '</span>' +
      '<span class="aud-badge">' + esc(info.codec) + '</span>' +
      ff + '</div>';

    var warns = "";
    if (info.warnings && info.warnings.length) {
      warns = '<div class="aud-warns">' +
        info.warnings.slice(0, 6).map(function (w) {
          return '<div class="aud-warn">' + esc(w) + '</div>';
        }).join("") + '</div>';
    }

    var replaceUI = "";
    if (info.replace_supported) {
      replaceUI =
        '<form id="audReplaceForm" class="aud-replace">' +
        '<div class="vp-insp-title">Replace (DEV only)</div>' +
        '<div class="dim aud-replace-help">Writes to the DEV data dir only — your live install is never touched. ' +
        'Upload a .wav (22050/mono/16 ideal) or .ogg.</div>' +
        '<input type="file" id="audReplaceFile" accept=".wav,.ogg,audio/wav,audio/ogg" />' +
        '<label class="aud-opt"><input type="checkbox" id="audNormalize" /> normalize to -1 dBFS</label>' +
        '<div class="aud-replace-btns">' +
        '<button id="audReplacePreview" class="ghost">preview</button>' +
        '<button id="audReplaceDeploy" class="primary">deploy to DEV</button>' +
        '</div>' +
        '<a id="audPreviewLink" class="ghost" style="display:none" download>download preview</a>' +
        '<div id="audReplaceStatus" class="aud-replace-status"></div>' +
        '</form>';
    } else {
      replaceUI = '<div class="vp-insp-section dim">Replace is not available for this container ' +
        '(only .pac and .ogg are replace targets; .sfd / .adx are read-only).</div>';
    }

    state.insp.innerHTML =
      '<div class="vp-insp-title">Audio</div>' +
      badge + warns +
      '<div class="vp-insp-section"><a class="ghost" href="' + esc(rawUrl(state.path)) +
      '" download>download raw</a></div>' +
      replaceUI;

    if (info.replace_supported) wireReplace(state, info);
  }

  function renderStage(state, info) {
    var records = info.records || [];
    var pickerHtml = "";
    if (info.decode_kind === "pac" && records.length > 1) {
      pickerHtml =
        '<div class="aud-picker"><label>record: ' +
        '<select id="audRecordSel">' +
        records.map(function (rec) {
          var label = "#" + rec.index +
            (rec.structured ? (" — " + fmtBytes(rec.pcm_bytes) + " · " + fmtDur(rec.duration_s))
                            : " — (opaque)");
          return '<option value="' + rec.index + '"' +
            (rec.structured ? "" : " disabled") + '>' + esc(label) + '</option>';
        }).join("") +
        '</select></label> <span class="dim">' + records.length + ' records</span></div>';
    }

    state.stage.innerHTML =
      '<div class="vp-stage-card aud-card">' +
      '<div class="vp-stage-card-title">' + esc(state.path) + '</div>' +
      pickerHtml +
      '<canvas id="audCanvas" width="720" height="140" class="aud-canvas"></canvas>' +
      '<audio id="audPlayer" controls preload="metadata" style="width:100%;margin-top:8px"></audio>' +
      '</div>';

    state.currentRecord = 0;
    var sel = state.stage.querySelector("#audRecordSel");
    if (sel) {
      // Default to the first structured record.
      var firstOk = records.find(function (r) { return r.structured; });
      if (firstOk) { sel.value = String(firstOk.index); state.currentRecord = firstOk.index; }
      sel.addEventListener("change", function () {
        state.currentRecord = parseInt(sel.value, 10) || 0;
        pointAudio(state, info, state.currentRecord);
      });
    }
    pointAudio(state, info, state.currentRecord);
  }

  // ---- perspective spec ----------------------------------------------
  window.PSOPerspectives.register("audio", {
    label: "Audio",
    match: function (entry, file) {
      if (entry && entry.category === "audio") return 100;
      var ext = ((file || "").split(".").pop() || "").toLowerCase();
      if (["ogg", "wav", "pac", "sfd", "mp3"].indexOf(ext) !== -1) return 90;
      return 0;
    },
    mount: function (stage, insp, ctx) {
      var path = ctx && ctx.path;
      var state = { stage: stage, insp: insp, path: path, currentRecord: 0 };
      stage._audioState = state;
      stage.innerHTML = '<div class="vp-stage-card"><div class="dim">loading audio…</div></div>';
      insp.innerHTML = '<div class="vp-insp-title">Audio</div><div class="dim">…</div>';
      if (!path) return;
      fetch(apiUrl("/api/audio/info", path), { cache: "no-store" })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (info) {
          state.info = info;
          renderStage(state, info);
          renderInspector(state, info);
        })
        .catch(function (e) {
          // Graceful fallback: a bare raw <audio> so the user can still play.
          // The HTTP-status detail is noise to the user (the file plays fine);
          // keep it in the console for debugging only.
          console.warn("[audio] info unavailable for", path, "-", e);
          stage.innerHTML =
            '<div class="vp-stage-card"><div class="vp-stage-card-title">' + esc(path) + '</div>' +
            '<audio id="audPlayer" controls preload="metadata" style="width:100%"></audio></div>';
          var a = stage.querySelector("#audPlayer");
          if (a) { a.src = rawUrl(path); a.play().catch(function () {}); }
          insp.innerHTML = '<div class="vp-insp-title">Audio</div>' +
            '<div class="vp-insp-section"><a class="ghost" href="' + esc(rawUrl(path)) +
            '" download>download raw</a></div>';
        });
    },
    unmount: function (stage) {
      var a = stage.querySelector("#audPlayer");
      if (a) { try { a.pause(); a.removeAttribute("src"); a.load(); } catch (_e) {} }
      if (stage._audioState) stage._audioState = null;
    },
  });
})();
