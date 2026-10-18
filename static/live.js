(function () {
  "use strict";
  var STORE = "ww:last_location";
  var form = document.getElementById("search-form");
  var dashRoot = document.getElementById("dash-root");
  var results = document.getElementById("live-results");
  if (!results) return;

  function $(id) { return document.getElementById(id); }
  function setText(id, value) { var e = $(id); if (e) e.textContent = value; }
  function show(id, on) { var e = $(id); if (e) e.hidden = !on; }
  function num(v, suffix, digits) {
    if (v === null || v === undefined) return "—";
    return (digits === undefined ? String(v) : Number(v).toFixed(digits)) + (suffix || "");
  }
  function store(value) { try { localStorage.setItem(STORE, value); } catch (e) { /* ignore */ } }
  function recall() { try { return localStorage.getItem(STORE) || ""; } catch (e) { return ""; } }

  function message(text, kind) {
    var box = $("live-message");
    if (!box) return;
    box.textContent = text || "";
    box.className = "notice " + (kind === "info" ? "notice-info" : "notice-warn");
    box.hidden = !text;
  }

  function minutes(seconds) { return Math.max(1, Math.round(seconds / 60)); }

  function renderStatus(data) {
    var d = data.data || {}, badge = $("data-status");
    var text, cls = "badge";
    if (d.status === "live") { text = "Live data"; cls += " badge-ok"; }
    else if (d.status === "cached") { text = "Cached · " + minutes(d.age_seconds) + " min old"; cls += " badge-info"; }
    else { text = "Stale data · " + minutes(d.age_seconds) + " min old"; cls += " badge-warn"; }
    badge.textContent = text;
    badge.className = cls;
    var note = $("data-note");
    note.hidden = d.status === "live" || d.status === "cached";
    note.textContent = d.note || "";
    setText("dash-freshness", text + " — retrieved " + new Date(d.fetched_at).toLocaleString() + " from " + (d.provider || "the weather provider") + ".");
  }

  function renderPrediction(p) {
    var isModel = p.method === "trained_model";
    setText("method-badge", p.method_label || (isModel ? "Trained ML model" : "Weather-based fallback"));
    $("method-badge").className = "badge " + (isModel ? "badge-ok" : "badge-info");
    setText("risk-level", p.risk_level + " risk");
    setText("risk-index", p.risk_index !== undefined && p.risk_index !== null
      ? p.risk_index + "/100 · " + (p.risk_index_label || "Weather risk index") : "");
    setText("risk-advice", p.advice || "");
    var panel = $("risk-panel"), key = String(p.risk_level).toLowerCase();
    panel.className = "risk-panel risk-" + (["low", "moderate", "high"].indexOf(key) >= 0 ? key : "moderate");

    var scores = $("model-scores");
    scores.replaceChildren();
    if (p.model_scores) {
      Object.keys(p.model_scores).forEach(function (c) {
        var chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = c + " score: " + Math.round(p.model_scores[c] * 100) + "%";
        scores.appendChild(chip);
      });
      scores.hidden = false;
    } else { scores.hidden = true; }

    var list = $("reasons");
    list.replaceChildren();
    (p.reasons || []).forEach(function (r) {
      var li = document.createElement("li");
      li.className = "chip";
      li.textContent = r;
      list.appendChild(li);
    });
    setText("factor-note", p.factor_note || (p.notes || []).join(" "));
    setText("method-explainer", isModel
      ? "Method: trained machine-learning model, version " + (p.model_version || "unknown") + ". " + (p.model_score_note || "")
      : "Method: transparent weather-based rule (not machine learning, not a probability).");

    var fb = $("fallback-note");
    fb.hidden = !p.fallback_message;
    fb.textContent = p.fallback_message || "";
  }

  function renderForecast(hourly) {
    var box = $("forecast-bars");
    box.replaceChildren();
    var rows = (hourly || []).filter(function (h) { return h.precipitation_mm !== null && h.precipitation_mm !== undefined; });
    show("forecast-empty", rows.length === 0);
    box.hidden = rows.length === 0;
    var max = Math.max.apply(null, rows.map(function (h) { return h.precipitation_mm; }).concat([1]));
    rows.forEach(function (h) {
      var col = document.createElement("div"); col.className = "bar-col";
      var val = document.createElement("span"); val.className = "bar-val"; val.textContent = h.precipitation_mm.toFixed(1) + " mm";
      var track = document.createElement("div"); track.className = "bar-track";
      var bar = document.createElement("div"); bar.className = "bar";
      bar.style.height = Math.max(3, (h.precipitation_mm / max) * 100) + "%";
      track.appendChild(bar);
      var label = document.createElement("span"); label.className = "bar-label"; label.textContent = String(h.time).slice(11, 16);
      var prob = document.createElement("span"); prob.className = "bar-prob";
      prob.textContent = h.probability_pct === null || h.probability_pct === undefined ? "" : Math.round(h.probability_pct) + "%";
      col.append(val, track, label, prob);
      box.appendChild(col);
    });
  }

  function render(data) {
    var place = data.location, w = data.weather;
    setText("place-name", place.name || "Selected location");
    setText("place-subtitle", [place.admin1, place.country].filter(Boolean).join(", "));
    renderStatus(data);
    renderPrediction(data.prediction);
    setText("temperature", num(w.temperature_c, " °C"));
    setText("humidity", num(w.humidity_pct, " %"));
    setText("precipitation", num(w.current_precipitation_mm, " mm"));
    setText("wind", num(w.wind_speed_kmh, " km/h"));
    setText("past-3h", num(w.past_3h_precipitation_mm, " mm"));
    setText("past-24h", num(w.past_24h_precipitation_mm, " mm"));
    setText("past-72h", num(w.past_72h_precipitation_mm, " mm"));
    setText("rain-probability", num(w.next_6h_rain_probability_pct, " %"));
    setText("next-rain", num(w.next_6h_precipitation_mm, " mm"));
    setText("observed-at", "Observed " + String(w.observed_at).replace("T", " ") + " (" + (w.timezone || "local time") + ")");
    renderForecast(data.hourly_next);
    var missing = (w.missing_fields || []).length;
    setText("disclaimer", (missing ? missing + " weather value(s) were unavailable from the provider and are shown as “—”. " : "") + (data.disclaimer || ""));
    show("live-results", true);
  }

  async function check(location) {
    message("");
    if (!navigator.onLine) { message("You appear to be offline. Live weather needs an internet connection."); return false; }
    show("live-loading", true);
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, 30000);
    try {
      var response = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ location: location }), signal: controller.signal
      });
      var data = null;
      try { data = await response.json(); } catch (e) { /* non-JSON error page */ }
      if (!response.ok) {
        var text = data && data.error && data.error.message ? data.error.message
          : data && typeof data.detail === "string" ? data.detail : "Unable to retrieve weather for this location.";
        throw new Error(text);
      }
      render(data);
      store(location);
      return true;
    } catch (error) {
      show("live-results", false);
      message(error.name === "AbortError" ? "The request took too long. Please try again."
        : error instanceof TypeError ? "Could not reach the server. Check your connection and try again."
        : error.message);
      return false;
    } finally {
      clearTimeout(timer);
      show("live-loading", false);
    }
  }

  if (form) {
    var input = $("location-input"), button = $("check-button");
    var last = recall();
    if (last) input.value = last;
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      var value = input.value.trim().replace(/\s+/g, " ");
      if (value.length < 2) { message("Please enter a city or area name (at least 2 characters)."); input.focus(); return; }
      button.disabled = true; button.textContent = "Checking…";
      var ok = await check(value);
      button.disabled = false; button.textContent = "Check risk";
      if (ok) results.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  if (dashRoot) {
    var saved = recall(), refresh = $("dash-refresh");
    if (!saved) { show("dash-empty", true); }
    else {
      refresh.hidden = false;
      refresh.addEventListener("click", function () { check(saved); });
      check(saved);
    }
  }
})();
