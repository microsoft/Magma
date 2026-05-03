const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld(
  "config",
  Object.freeze({
    baseApiUrl: process.env.DOME_BACKEND_URL || "https://wafair-dome.hf.space",
    agentApiUrl: process.env.DOME_AGENT_URL || "http://127.0.0.1:8010",
  })
);
