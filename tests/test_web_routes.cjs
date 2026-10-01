const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

function loadRoutes(origin) {
  const source = fs.readFileSync(path.join(__dirname, "../web/mgpu_orchestrator.js"), "utf8")
    .replace(/^import .*;\r?\n/gm, "")
    .replaceAll("import.meta.url", JSON.stringify(`${origin}/extensions/mgpu/mgpu_orchestrator.js`));
  const context = vm.createContext({
    URL, Request,
    app: { registerExtension() {} },
    api: { api_base: "", apiURL: (route) => `${origin}/api${route}` },
    window: { location: { href: `${origin}/`, origin } },
  });
  vm.runInContext(source, context);
  return context;
}

test("single and batch cancel map through both API route forms", () => {
  const routes = loadRoutes("http://localhost:8188");
  for (const prefix of ["", "/api"]) {
    for (const suffix of ["cancel", "job-id/cancel", "job-id/cancel?clientId=a"]) {
      assert.equal(routes.mappedMgpuRoute(`${prefix}/jobs/${suffix}`, "POST"), `/mgpu/jobs/${suffix}`);
    }
  }
  for (const route of ["/jobs/id/retry", "/jobs/cancel/extra", "/jobs/id/cancelled", "/mgpu/jobs/cancel"]) {
    assert.equal(routes.mappedMgpuRoute(route, "POST"), null);
  }
  assert.equal(routes.mappedMgpuRoute("/interrupt", "POST"), "/mgpu/interrupt");
  assert.equal(routes.mappedMgpuRoute("/queue", "POST"), "/mgpu/queue");
});

test("direct fetch cancel routing is independent of tunnel, IP, or localhost", async () => {
  for (const origin of ["https://example.trycloudflare.com", "http://192.0.2.1:8188", "http://localhost:8188"]) {
    const routes = loadRoutes(origin);
    const input = new Request(`${origin}/api/jobs/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_ids: ["job-a", "job-b"] }),
    });
    const route = routes.routeFromSameOriginApiUrl(input.url);
    const mapped = routes.mappedMgpuRoute(route, input.method);
    const [rewritten] = routes.rewriteFetchCall(input, undefined, mapped);
    assert.equal(rewritten.url, `${origin}/api/mgpu/jobs/cancel`);
    assert.equal(rewritten.method, "POST");
    assert.equal(rewritten.headers.get("Content-Type"), "application/json");
    assert.deepEqual(await rewritten.json(), { job_ids: ["job-a", "job-b"] });
    assert.equal(routes.routeFromSameOriginApiUrl("https://other.example/api/jobs/cancel"), null);
  }
});

test("worker cancellation failures are returned without falling back to the parent queue", async () => {
  const routes = loadRoutes("http://localhost:8188");
  const response = { status: 502 };
  const calls = [];
  const fetch = async (...args) => { calls.push(args); return response; };
  assert.equal(await routes.fetchWithFallback(fetch, "/jobs/cancel", "/mgpu/jobs/cancel", { method: "POST" }), response);
  assert.equal(calls.length, 1);
});
