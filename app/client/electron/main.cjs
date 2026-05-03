const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

const BACKEND_URL = process.env.DOME_BACKEND_URL || "https://wafair-dome.hf.space";
const AGENT_URL = process.env.DOME_AGENT_URL || "http://127.0.0.1:8010";
const AGENT_START_TIMEOUT_MS = Number(process.env.DOME_AGENT_START_TIMEOUT_MS || 120000);

let mainWindow = null;
let agentProcess = null;
let ownsAgentProcess = false;
let appIsLoading = false;
let agentStartupPromise = null;

function parseAgentUrl() {
  const url = new URL(AGENT_URL);
  return {
    host: url.hostname || "127.0.0.1",
    port: url.port || "8010",
  };
}

function getAgentExecutableName() {
  return process.platform === "win32" ? "dome-agent.exe" : "dome-agent";
}

function getAgentPath() {
  const executable = getAgentExecutableName();
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "agent", executable);
  }
  return path.resolve(__dirname, "..", "..", "dist", executable);
}

function checkAgentHealth(timeoutMs = 1000) {
  return new Promise((resolve) => {
    const req = http.get(`${AGENT_URL}/health`, { timeout: timeoutMs }, (res) => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
    req.on("error", () => resolve(false));
  });
}

async function waitForAgent(timeoutMs = AGENT_START_TIMEOUT_MS) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await checkAgentHealth(750)) return true;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function startupHtml({ title, message, detail, isError = false }) {
  const safeTitle = escapeHtml(title);
  const safeMessage = escapeHtml(message);
  const safeDetail = detail ? escapeHtml(detail) : "";

  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${safeTitle}</title>
    <style>
      :root {
        color-scheme: light;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f5f7fb;
        color: #172033;
      }
      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
      }
      main {
        width: min(520px, calc(100vw - 48px));
        text-align: center;
      }
      .spinner {
        width: 34px;
        height: 34px;
        margin: 0 auto 22px;
        border: 3px solid #d9e2ef;
        border-top-color: #1677ff;
        border-radius: 50%;
        animation: spin 0.9s linear infinite;
      }
      .error {
        width: 40px;
        height: 40px;
        margin: 0 auto 20px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: #fff1f0;
        color: #c41d7f;
        font-size: 28px;
        font-weight: 700;
      }
      h1 {
        margin: 0 0 12px;
        font-size: 24px;
        font-weight: 700;
      }
      p {
        margin: 0;
        font-size: 15px;
        line-height: 1.55;
        color: #536070;
      }
      code {
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 13px;
      }
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
    </style>
  </head>
  <body>
    <main>
      ${isError ? '<div class="error">!</div>' : '<div class="spinner"></div>'}
      <h1>${safeTitle}</h1>
      <p>${safeMessage}</p>
      ${safeDetail ? `<p style="margin-top: 12px;"><code>${safeDetail}</code></p>` : ""}
    </main>
  </body>
</html>`;
}

function loadStartupPage() {
  if (!mainWindow) return;
  mainWindow.loadURL(
    `data:text/html;charset=utf-8,${encodeURIComponent(
      startupHtml({
        title: "Starting local agent",
        message: "The desktop app will open when the local automation agent is ready.",
        detail: AGENT_URL,
      })
    )}`
  );
}

function loadAgentErrorPage(message) {
  if (!mainWindow) return;
  mainWindow.loadURL(
    `data:text/html;charset=utf-8,${encodeURIComponent(
      startupHtml({
        title: "Local agent is not ready",
        message,
        detail: AGENT_URL,
        isError: true,
      })
    )}`
  );
}

function loadApp() {
  if (!mainWindow || appIsLoading) return;
  appIsLoading = true;

  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  } else {
    mainWindow.loadURL("http://127.0.0.1:4000");
  }
}

async function startAgent() {
  if (await checkAgentHealth()) {
    console.log(`[agent] Reusing existing agent at ${AGENT_URL}`);
    return true;
  }

  const { host, port } = parseAgentUrl();
  const agentPath = getAgentPath();

  agentProcess = spawn(
    agentPath,
    ["--server-url", BACKEND_URL, "--host", host, "--port", port],
    {
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    }
  );
  ownsAgentProcess = true;

  agentProcess.stdout.on("data", (data) => {
    console.log(`[agent] ${data.toString().trimEnd()}`);
  });
  agentProcess.stderr.on("data", (data) => {
    console.error(`[agent] ${data.toString().trimEnd()}`);
  });
  agentProcess.on("exit", (code, signal) => {
    if (ownsAgentProcess) {
      console.log(`[agent] exited code=${code} signal=${signal}`);
    }
    agentProcess = null;
  });
  agentProcess.on("error", (error) => {
    loadAgentErrorPage(
      `Could not start the local agent. Expected executable at ${agentPath}. ${error.message}`
    );
  });

  return waitForAgent();
}

async function ensureAgentThenLoadApp() {
  if (!agentStartupPromise) {
    agentStartupPromise = startAgent();
  }

  const ready = await agentStartupPromise;
  if (!mainWindow) return;

  if (!ready) {
    agentStartupPromise = null;
    loadAgentErrorPage(
      "The local automation agent did not respond before the startup timeout."
    );
    return;
  }

  loadApp();
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  loadStartupPage();

  mainWindow.on("closed", () => {
    mainWindow = null;
    appIsLoading = false;
  });
}

function stopAgent() {
  if (agentProcess && ownsAgentProcess) {
    ownsAgentProcess = false;
    agentProcess.kill();
  }
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    createWindow();
    ensureAgentThenLoadApp();
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
      ensureAgentThenLoadApp();
    }
  });

  app.on("before-quit", stopAgent);

  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });
}
