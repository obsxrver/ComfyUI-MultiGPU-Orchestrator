import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "comfyui.mgpu.orchestrator";
const FALLBACK_STATUSES = new Set([404, 424, 503]);
let warnedFallback = false;
let bypassDirectFetchRewrite = false;

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
