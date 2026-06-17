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

    originalFetchApi("/mgpu/status")
      .then(async (response) => {
        if (!response.ok) {
          warnFallback(`Status endpoint returned ${response.status}.`);
          return;
        }
        const status = await response.json();
        const healthyWorkers = (status.workers || []).filter(
          (worker) => worker.status === "healthy",
        );
        if (healthyWorkers.length === 0) {
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
