/// <reference types="vite/client" />

interface WindowConfig {
  baseApiUrl: string;
  agentApiUrl?: string;
}

declare global {
  interface Window {
    config: WindowConfig;
  }
}

export {};
