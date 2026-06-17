import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "comfyui.mgpu.orchestrator";
const FALLBACK_STATUSES = new Set([404, 424, 503]);
let warnedFallback = false;
let bypassDirectFetchRewrite = false;
const rowProgressByPrompt = new Map();
let rowProgressFrame = 0;

function normalizeRoute(route) {
  return String(route || "");
}

function methodOf(options) {
  return String(options?.method || "GET").toUpperCase();
}

function routeToMgpu(normalized, fromPrefix, toPrefix) {
  return `${toPrefix}${normalized.slice(fromPrefix.length)}`;
}

function comparableRoute(route) {
  return route.startsWith("/api/") ? route.slice(4) : route;
}

function mappedMgpuRoute(route, method) {
  const comparable = comparableRoute(route);

  if (comparable.startsWith("/mgpu/")) {
    return null;
  }

  if (method === "POST" && comparable === "/prompt") {
    return "/mgpu/prompt";
  }

  if (method === "GET" && comparable === "/prompt") {
    return "/mgpu/prompt";
  }

  if (method === "POST" && comparable === "/interrupt") {
    return "/mgpu/interrupt";
  }

  if (method === "POST" && comparable === "/free") {
    return "/mgpu/free";
  }

  if (method === "POST" && comparable === "/queue") {
    return "/mgpu/queue";
  }

  if (method === "POST" && comparable === "/history") {
    return "/mgpu/history";
  }

  if (
    method === "POST" &&
    (comparable === "/assets/seed" ||
      comparable.startsWith("/assets/seed?") ||
      comparable === "/assets/seed/cancel" ||
      comparable.startsWith("/assets/seed/cancel?"))
  ) {
    return routeToMgpu(comparable, "/assets", "/mgpu/assets");
  }

  if (method === "GET" && (comparable === "/jobs" || comparable.startsWith("/jobs?"))) {
    return routeToMgpu(comparable, "/jobs", "/mgpu/jobs");
  }

  if (method === "GET" && comparable.startsWith("/jobs/")) {
    return routeToMgpu(comparable, "/jobs", "/mgpu/jobs");
  }

  if (method === "GET" && comparable === "/queue") {
    return "/mgpu/queue";
  }

  if (method === "GET" && (comparable === "/history" || comparable.startsWith("/history"))) {
    return routeToMgpu(comparable, "/history", "/mgpu/history");
  }

  if (method === "GET" && (comparable === "/assets" || comparable.startsWith("/assets?"))) {
    return routeToMgpu(comparable, "/assets", "/mgpu/assets");
  }

  if ((method === "GET" || method === "HEAD") && comparable.startsWith("/assets/")) {
    return routeToMgpu(comparable, "/assets", "/mgpu/assets");
  }

  if (method === "GET" && (comparable === "/tags" || comparable.startsWith("/tags?"))) {
    return routeToMgpu(comparable, "/tags", "/mgpu/tags");
  }

  return null;
}

function routeFromSameOriginApiUrl(url) {
  const parsed = new URL(url, window.location.href);
  if (parsed.origin !== window.location.origin) {
    return null;
  }

  const apiPrefix = `${api.api_base || ""}/api`;
  if (!parsed.pathname.startsWith(apiPrefix)) {
    return null;
  }

  const route = parsed.pathname.slice(apiPrefix.length) + parsed.search;
  return route.startsWith("/") ? route : `/${route}`;
}

function rewriteFetchCall(input, init, mappedRoute) {
  const rewrittenUrl = api.apiURL(mappedRoute);
  if (input instanceof Request) {
    const merged = init ? new Request(input, init) : input.clone();
    const requestInit = {
      method: merged.method,
      headers: merged.headers,
      mode: merged.mode,
      credentials: merged.credentials,
      cache: merged.cache,
      redirect: merged.redirect,
      referrer: merged.referrer,
      referrerPolicy: merged.referrerPolicy,
      integrity: merged.integrity,
      keepalive: merged.keepalive,
      signal: merged.signal,
    };
    if (merged.method !== "GET" && merged.method !== "HEAD") {
      requestInit.body = merged.body;
      requestInit.duplex = "half";
    }
    return [new Request(rewrittenUrl, requestInit), undefined];
  }
  if (input instanceof URL) {
    return [new URL(rewrittenUrl, window.location.href), init];
  }
  return [rewrittenUrl, init];
}

function warnFallback(message) {
  if (warnedFallback) return;
  warnedFallback = true;
  console.warn(`[ComfyUI-MGPU] ${message}`);
  try {
    app?.ui?.dialog?.show?.(`[ComfyUI-MGPU] ${message}`);
  } catch (_error) {
    // Older/newer ComfyUI frontends expose different notification APIs.
  }
}

function clampPercent(value) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return 0;
  return Math.min(Math.max(Math.round(numberValue), 0), 100);
}

function formatPercent(value) {
  return `${clampPercent(value)}%`;
}

function summarizeProgressState(data) {
  const promptId = data?.prompt_id;
  if (!promptId) return null;

  const mgpu = data.mgpu || {};
  const nodes = Object.values(data.nodes || {}).filter(
    (node) => node && typeof node === "object",
  );
  const runningNode = nodes.find((node) => node.state === "running");
  const finishedCount = nodes.filter((node) => node.state === "finished").length;
  const currentValue = Number(runningNode?.value || 0);
  const currentMax = Number(runningNode?.max || 0);
  const currentRatio = currentMax > 0 ? currentValue / currentMax : 0;
  const totalNodes = Number(mgpu.total_nodes || nodes.length || 1);
  const totalPercent =
    mgpu.total_percent ?? ((finishedCount + currentRatio) / totalNodes) * 100;

  return {
    promptId: String(promptId),
    totalPercent: clampPercent(totalPercent),
    currentNodePercent: clampPercent(mgpu.current_node_percent ?? currentRatio * 100),
    currentNodeLabel:
      mgpu.current_node_label ||
      runningNode?.node_label ||
      runningNode?.node_type ||
      runningNode?.real_node_id ||
      runningNode?.node_id ||
      "",
  };
}

function ensureSecondaryText(row, textColumn) {
  let wrapper = row.querySelector("[data-mgpu-secondary]");
  if (!wrapper) {
    wrapper = document.createElement("div");
    wrapper.dataset.mgpuSecondary = "true";
    wrapper.className = "min-w-0 text-xs leading-none text-text-secondary";
    const span = document.createElement("span");
    span.className = "block truncate";
    wrapper.appendChild(span);
    textColumn.appendChild(wrapper);
  }
  return wrapper.querySelector("span");
}

function ensureProgressBars(row) {
  const item = row.firstElementChild;
  if (!item) return null;
  let bar = item.querySelector(":scope > .mgpu-row-progress");
  if (!bar) {
    bar = document.createElement("div");
    bar.className = "mgpu-row-progress";
    bar.innerHTML = '<div data-mgpu-total></div><div data-mgpu-current></div>';
    item.insertBefore(bar, item.firstChild);
  }
  return bar;
}

function clearInjectedRowProgress(row) {
  row.querySelector(".mgpu-row-progress")?.remove();
  row.querySelector("[data-mgpu-secondary]")?.remove();
}

function updateQueueProgressRowsNow() {
  rowProgressFrame = 0;
  document.querySelectorAll("[data-job-id]").forEach((row) => {
    const progress = rowProgressByPrompt.get(String(row.dataset.jobId || ""));
    if (!progress) {
      clearInjectedRowProgress(row);
      return;
    }

    const spans = row.querySelectorAll("span.block.truncate");
    const primary = spans[0];
    const textColumn = primary?.closest(".flex-col");
    if (!primary || !textColumn) return;

    const secondary = spans[1] || ensureSecondaryText(row, textColumn);
    const primaryText = `Total: ${formatPercent(progress.totalPercent)}`;
    const secondaryText = progress.currentNodeLabel
      ? `${progress.currentNodeLabel}: ${formatPercent(progress.currentNodePercent)}`
      : "";

    primary.textContent = primaryText;
    primary.title = primaryText;
    if (secondary) {
      secondary.textContent = secondaryText;
      secondary.title = secondaryText;
    }

    const bar = ensureProgressBars(row);
    if (bar) {
      const total = bar.querySelector("[data-mgpu-total]");
      const current = bar.querySelector("[data-mgpu-current]");
      if (total) total.style.width = `${progress.totalPercent}%`;
      if (current) current.style.width = `${progress.currentNodePercent}%`;
    }
  });
}

function scheduleQueueProgressRowsUpdate() {
  if (rowProgressFrame) return;
  rowProgressFrame = requestAnimationFrame(updateQueueProgressRowsNow);
}

function installQueueProgressRowPatch() {
  if (api.__mgpuQueueProgressRowsPatched) return;
  api.__mgpuQueueProgressRowsPatched = true;

  const style = document.createElement("style");
  style.textContent = `
    .mgpu-row-progress {
      position: absolute;
      inset: 0;
      z-index: 0;
      overflow: hidden;
      border-radius: inherit;
      pointer-events: none;
    }
    .mgpu-row-progress > div {
      position: absolute;
      inset-block: 0;
      left: 0;
      transition: width 160ms ease;
    }
    .mgpu-row-progress > [data-mgpu-total] {
      background: var(--color-interface-panel-job-progress-primary, rgba(21, 101, 192, 0.75));
    }
    .mgpu-row-progress > [data-mgpu-current] {
      background: var(--color-interface-panel-job-progress-secondary, rgba(30, 136, 229, 0.45));
    }
  `;
  document.head.appendChild(style);

  api.addEventListener("progress_state", (event) => {
    const progress = summarizeProgressState(event.detail);
    if (!progress) return;
    rowProgressByPrompt.set(progress.promptId, progress);
    scheduleQueueProgressRowsUpdate();
  });

  for (const eventName of ["execution_success", "execution_error", "execution_interrupted"]) {
    api.addEventListener(eventName, (event) => {
      const promptId = event.detail?.prompt_id;
      if (promptId) rowProgressByPrompt.delete(String(promptId));
      scheduleQueueProgressRowsUpdate();
    });
  }

  const observer = new MutationObserver(scheduleQueueProgressRowsUpdate);
  observer.observe(document.body, { childList: true, subtree: true });
}

function gpuIconSvg() {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M4 7.5A2.5 2.5 0 0 1 6.5 5h8A2.5 2.5 0 0 1 17 7.5V8h1.75A1.25 1.25 0 0 1 20 9.25V11h1.25a.75.75 0 0 1 0 1.5H20v2.25A1.25 1.25 0 0 1 18.75 16H17v.5a2.5 2.5 0 0 1-2.5 2.5h-8A2.5 2.5 0 0 1 4 16.5V16H2.75a.75.75 0 0 1 0-1.5H4v-2H2.75a.75.75 0 0 1 0-1.5H4V9H2.75a.75.75 0 0 1 0-1.5H4Zm2.5-1A1 1 0 0 0 5.5 7.5v9a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1h-8Zm10.5 3V14.5h1.5V9.5H17Z"/>
      <path d="M8.25 9.25h4.5a.75.75 0 0 1 .75.75v4a.75.75 0 0 1-.75.75h-4.5A.75.75 0 0 1 7.5 14v-4a.75.75 0 0 1 .75-.75Zm.75 1.5v2.5h3v-2.5H9Z"/>
    </svg>
  `;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function installMultiGpuMenu(originalFetchApi) {
  if (api.__mgpuSidebarTabRegistered) return;
  api.__mgpuSidebarTabRegistered = true;
  const manager = app?.extensionManager;
  if (!manager?.registerSidebarTab) {
    api.__mgpuSidebarTabRegistered = false;
    setTimeout(() => installMultiGpuMenu(originalFetchApi), 500);
    return;
  }
  const iconUrl = new URL("./mgpu-icon.svg", import.meta.url).href;

  const state = {
    panel: null,
    refreshTimer: 0,
    loading: false,
    actionKey: "",
    status: null,
    error: "",
    autoStart: true,
  };

  if (!document.getElementById("mgpu-sidebar-tab-style")) {
    const style = document.createElement("style");
    style.id = "mgpu-sidebar-tab-style";
    style.textContent = `
    .mgpu-gpu-icon {
      display: inline-block;
      width: var(--sidebar-icon-size, 1rem);
      height: var(--sidebar-icon-size, 1rem);
      background-color: currentColor;
      -webkit-mask: url("${iconUrl}") center / contain no-repeat;
      mask: url("${iconUrl}") center / contain no-repeat;
    }
    .mgpu-extension-slot {
      min-height: 100%;
    }
    #mgpu-menu-panel {
      min-height: 100%;
      display: flex;
      flex-direction: column;
      color: var(--fg-color, #ededed);
      background: var(--comfy-menu-bg, var(--bg-color, #181818));
      font-family: var(--font-family, system-ui, sans-serif);
    }
    .mgpu-menu-header,
    .mgpu-menu-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--interface-stroke, rgba(255, 255, 255, 0.09));
    }
    .mgpu-menu-footer {
      margin-top: auto;
      border-top: 1px solid var(--interface-stroke, rgba(255, 255, 255, 0.09));
      border-bottom: 0;
    }
    .mgpu-menu-title {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      font-size: 14px;
      font-weight: 650;
    }
    .mgpu-menu-title svg {
      width: 19px;
      height: 19px;
      fill: currentColor;
    }
    .mgpu-icon-button {
      width: 30px;
      height: 30px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: inherit;
      background: transparent;
      border: 1px solid var(--interface-stroke, rgba(255, 255, 255, 0.12));
      border-radius: 6px;
      cursor: pointer;
    }
    .mgpu-icon-button:hover {
      background: var(--interface-panel-hover-surface, rgba(255, 255, 255, 0.08));
    }
    .mgpu-menu-body {
      overflow: auto;
      padding: 8px;
    }
    .mgpu-worker-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 10px;
      border-radius: 6px;
    }
    .mgpu-worker-row + .mgpu-worker-row {
      border-top: 1px solid var(--interface-stroke, rgba(255, 255, 255, 0.08));
    }
    .mgpu-worker-main {
      display: grid;
      gap: 7px;
      min-width: 0;
    }
    .mgpu-worker-name {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      font-size: 13px;
      font-weight: 650;
    }
    .mgpu-worker-url {
      overflow: hidden;
      color: var(--input-text, #a6a6a6);
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
    }
    .mgpu-worker-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .mgpu-pill,
    .mgpu-status {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 0 7px;
      border-radius: 999px;
      font-size: 11px;
      line-height: 1;
      background: rgba(255, 255, 255, 0.08);
      color: var(--input-text, #c9c9c9);
    }
    .mgpu-status[data-tone="healthy"] {
      background: rgba(32, 164, 97, 0.18);
      color: #72dfa8;
    }
    .mgpu-status[data-tone="starting"] {
      background: rgba(217, 160, 43, 0.18);
      color: #f2c66d;
    }
    .mgpu-status[data-tone="unhealthy"] {
      background: rgba(224, 78, 78, 0.18);
      color: #ff9a9a;
    }
    .mgpu-worker-action {
      min-width: 72px;
      height: 30px;
      padding: 0 10px;
      color: var(--fg-color, #f2f2f2);
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--interface-stroke, rgba(255, 255, 255, 0.13));
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
    }
    .mgpu-worker-action:hover {
      background: var(--interface-panel-hover-surface, rgba(255, 255, 255, 0.13));
    }
    .mgpu-worker-action:disabled,
    .mgpu-icon-button:disabled {
      opacity: 0.5;
      cursor: wait;
    }
    .mgpu-empty,
    .mgpu-error {
      padding: 18px 12px;
      color: var(--input-text, #b7b7b7);
      font-size: 12px;
      line-height: 1.4;
    }
    .mgpu-error {
      color: #ff9a9a;
    }
    .mgpu-toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      width: 100%;
      font-size: 12px;
      cursor: pointer;
    }
    .mgpu-switch {
      position: relative;
      width: 38px;
      height: 22px;
      flex: 0 0 auto;
    }
    .mgpu-switch input {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }
    .mgpu-switch span {
      position: absolute;
      inset: 0;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.18);
      transition: background 140ms ease;
    }
    .mgpu-switch span::after {
      content: "";
      position: absolute;
      width: 16px;
      height: 16px;
      top: 3px;
      left: 3px;
      border-radius: 50%;
      background: #f4f4f4;
      transition: transform 140ms ease;
    }
    .mgpu-switch input:checked + span {
      background: #2f8f69;
    }
    .mgpu-switch input:checked + span::after {
      transform: translateX(16px);
    }
    `;
    document.head.appendChild(style);
  }

  function statusTone(status) {
    if (status === "healthy") return "healthy";
    if (status === "starting" || status === "new") return "starting";
    return "unhealthy";
  }

  function renderPanel() {
    if (!state.panel) return;
    const workers = Array.isArray(state.status?.workers) ? state.status.workers : [];
    state.panel.innerHTML = `
      <div class="mgpu-menu-header">
        <div class="mgpu-menu-title">${gpuIconSvg()}<span>MultiGPU</span></div>
        <button class="mgpu-icon-button" type="button" data-mgpu-refresh title="Refresh" ${state.loading ? "disabled" : ""}>&#8635;</button>
      </div>
      <div class="mgpu-menu-body"></div>
      <div class="mgpu-menu-footer">
        <label class="mgpu-toggle">
          <span>Auto start at startup</span>
          <span class="mgpu-switch">
            <input type="checkbox" data-mgpu-auto-start ${state.autoStart ? "checked" : ""}>
            <span></span>
          </span>
        </label>
      </div>
    `;

    const body = state.panel.querySelector(".mgpu-menu-body");
    if (state.error) {
      body.innerHTML = `<div class="mgpu-error">${state.error}</div>`;
    } else if (state.loading && !state.status) {
      body.innerHTML = '<div class="mgpu-empty">Loading...</div>';
    } else if (workers.length === 0) {
      body.innerHTML = '<div class="mgpu-empty">No workers are running.</div>';
    } else {
      body.innerHTML = workers
        .map((worker) => {
          const status = String(worker.status || "unknown");
          const gpuIndex = Number(worker.gpu_index);
          const workerUrl = escapeHtml(worker.url || "");
          const isHealthy = status === "healthy";
          const action = isHealthy ? "stop" : "restart";
          const actionLabel = isHealthy ? "Stop" : "Restart";
          const actionKey = `${action}:${gpuIndex}`;
          const busy = state.actionKey === actionKey;
          const disabled = status === "starting" || busy;
          return `
            <div class="mgpu-worker-row">
              <div class="mgpu-worker-main">
                <div class="mgpu-worker-name">
                  <span>GPU ${gpuIndex}</span>
                  <span class="mgpu-status" data-tone="${statusTone(status)}">${escapeHtml(status)}</span>
                </div>
                <div class="mgpu-worker-stats">
                  <span class="mgpu-pill">Active: ${Number(worker.running || 0)}</span>
                  <span class="mgpu-pill">Pending: ${Number(worker.pending || 0)}</span>
                  ${worker.pid ? `<span class="mgpu-pill">PID: ${worker.pid}</span>` : ""}
                </div>
                <div class="mgpu-worker-url" title="${workerUrl}">${workerUrl}</div>
              </div>
              <button class="mgpu-worker-action" type="button" data-mgpu-action="${action}" data-gpu-index="${gpuIndex}" ${disabled ? "disabled" : ""}>
                ${busy ? "..." : actionLabel}
              </button>
            </div>
          `;
        })
        .join("");
    }

    state.panel.querySelector("[data-mgpu-refresh]")?.addEventListener("click", refreshStatus);
    state.panel.querySelector("[data-mgpu-auto-start]")?.addEventListener("change", updateAutoStart);
    state.panel.querySelectorAll("[data-mgpu-action]").forEach((button) => {
      button.addEventListener("click", () => runWorkerAction(button.dataset.gpuIndex, button.dataset.mgpuAction));
    });
  }

  async function refreshStatus() {
    state.loading = true;
    state.error = "";
    renderPanel();
    try {
      const response = await originalFetchApi("/mgpu/status");
      if (!response.ok) throw new Error(`status ${response.status}`);
      const status = await response.json();
      state.status = status;
      state.autoStart = status.auto_start !== false;
    } catch (error) {
      state.error = `Status unavailable: ${error}`;
    } finally {
      state.loading = false;
      renderPanel();
    }
  }

  async function runWorkerAction(gpuIndex, action) {
    const normalizedAction = action === "stop" ? "stop" : "restart";
    state.actionKey = `${normalizedAction}:${gpuIndex}`;
    state.error = "";
    renderPanel();
    try {
      const response = await originalFetchApi(`/mgpu/workers/${gpuIndex}/${normalizedAction}`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(`${normalizedAction} failed with status ${response.status}`);
      await refreshStatus();
    } catch (error) {
      state.error = String(error);
      renderPanel();
    } finally {
      state.actionKey = "";
      renderPanel();
    }
  }

  async function updateAutoStart(event) {
    const checked = Boolean(event.target.checked);
    state.autoStart = checked;
    renderPanel();
    try {
      const response = await originalFetchApi("/mgpu/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auto_start: checked }),
      });
      if (!response.ok) throw new Error(`settings ${response.status}`);
      const settings = await response.json();
      state.autoStart = settings.auto_start !== false;
    } catch (error) {
      state.autoStart = !checked;
      state.error = `Setting update failed: ${error}`;
    }
    renderPanel();
  }

  function destroyPanel() {
    state.panel = null;
    if (state.refreshTimer) {
      clearInterval(state.refreshTimer);
      state.refreshTimer = 0;
    }
  }

  function mountPanel(container) {
    destroyPanel();
    container.classList.add("mgpu-extension-slot");
    container.innerHTML = '<div id="mgpu-menu-panel"></div>';
    state.panel = container.querySelector("#mgpu-menu-panel");
    renderPanel();
    refreshStatus();
    state.refreshTimer = setInterval(refreshStatus, 5000);
  }

  function registerOfficialSidebarTab() {
    const existingTabs = manager.getSidebarTabs?.() || [];
    if (existingTabs.some((tab) => tab?.id === "mgpu")) return;
    manager.registerSidebarTab({
      id: "mgpu",
      icon: "mgpu-gpu-icon",
      title: "MultiGPU",
      label: "MultiGPU",
      tooltip: "MultiGPU",
      type: "custom",
      render: mountPanel,
      destroy: destroyPanel,
    });
  }

  registerOfficialSidebarTab();
}

async function fetchWithFallback(originalFetchApi, originalRoute, mgpuRoute, options) {
  try {
    const response = await originalFetchApi.call(api, mgpuRoute, options);
    if (!FALLBACK_STATUSES.has(response.status)) {
      return response;
    }
    const detail = await response.clone().text().catch(() => "");
    warnFallback(
      `Multi-GPU routing is unavailable; falling back to native ${originalRoute}. ${detail}`,
    );
  } catch (error) {
    warnFallback(
      `Multi-GPU routing failed; falling back to native ${originalRoute}. ${error}`,
    );
  }
  bypassDirectFetchRewrite = true;
  try {
    return await originalFetchApi.call(api, originalRoute, options);
  } finally {
    bypassDirectFetchRewrite = false;
  }
}

app.registerExtension({
  name: EXTENSION_NAME,
  async setup() {
    if (api.__mgpuOrchestratorWrapped) return;

    const originalFetchApi = api.fetchApi.bind(api);
    api.fetchApi = async function patchedFetchApi(route, options) {
      const normalized = normalizeRoute(route);
      const method = methodOf(options);
      const mgpuRoute = mappedMgpuRoute(normalized, method);

      if (mgpuRoute) {
        return fetchWithFallback(originalFetchApi, normalized, mgpuRoute, options);
      }

      return originalFetchApi(route, options);
    };

    const originalFetch = window.fetch.bind(window);
    window.fetch = function patchedFetch(input, init) {
      if (bypassDirectFetchRewrite) {
        return originalFetch(input, init);
      }
      const requestUrl =
        typeof input === "string" || input instanceof URL ? input.toString() : input?.url;
      const originalRoute = requestUrl ? routeFromSameOriginApiUrl(requestUrl) : null;
      const method = String(init?.method || input?.method || "GET").toUpperCase();
      const mgpuRoute = originalRoute ? mappedMgpuRoute(originalRoute, method) : null;
      if (mgpuRoute) {
        const [rewrittenInput, rewrittenInit] = rewriteFetchCall(input, init, mgpuRoute);
        return originalFetch(rewrittenInput, rewrittenInit);
      }
      return originalFetch(input, init);
    };

    api.__mgpuOrchestratorWrapped = true;
    installQueueProgressRowPatch();
    installMultiGpuMenu(originalFetchApi);

    originalFetchApi("/mgpu/status")
      .then(async (response) => {
        if (!response.ok) {
          warnFallback(`Status endpoint returned ${response.status}.`);
          return;
        }
        const status = await response.json();
        if (!status.started) {
          console.info("[ComfyUI-MGPU] Worker startup is not active.");
          return;
        }
        const healthyWorkers = (status.workers || []).filter(
          (worker) => worker.status === "healthy",
        );
        if ((status.workers || []).length > 0 && healthyWorkers.length === 0) {
          warnFallback("No healthy workers are ready yet.");
        } else {
          console.info(
            `[ComfyUI-MGPU] Ready with ${healthyWorkers.length} worker(s).`,
          );
        }
      })
      .catch((error) => {
        warnFallback(`Status endpoint is unavailable. ${error}`);
      });
  },
});
