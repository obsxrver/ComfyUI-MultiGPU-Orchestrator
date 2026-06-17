import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "comfyui.mgpu.orchestrator";
const FALLBACK_STATUSES = new Set([404, 424, 503]);
let warnedFallback = false;

function normalizeRoute(route) {
  return String(route || "");
}

function methodOf(options) {
  return String(options?.method || "GET").toUpperCase();
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
  return originalFetchApi.call(api, originalRoute, options);
}

app.registerExtension({
  name: EXTENSION_NAME,
  async setup() {
    if (api.__mgpuOrchestratorWrapped) return;

    const originalFetchApi = api.fetchApi.bind(api);
    api.fetchApi = async function patchedFetchApi(route, options) {
      const normalized = normalizeRoute(route);
      const method = methodOf(options);

      if (method === "POST" && normalized === "/prompt") {
        return fetchWithFallback(originalFetchApi, normalized, "/mgpu/prompt", options);
      }

      if (method === "POST" && normalized === "/interrupt") {
        return fetchWithFallback(originalFetchApi, normalized, "/mgpu/interrupt", options);
      }

      if (method === "POST" && normalized === "/free") {
        return fetchWithFallback(originalFetchApi, normalized, "/mgpu/free", options);
      }

      return originalFetchApi(route, options);
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
