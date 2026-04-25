import axios from "axios";
import type { Task } from "../types/task";
import type { ApiResult } from "../types/api";

export interface AgentDispatchResponse {
  task_id: string;
  status: "accepted";
}

const getAgentBaseUrl = () =>
  (
    import.meta.env.VITE_AGENT_URL ||
    window.config?.agentApiUrl ||
    "http://localhost:8010"
  ).replace(/\/$/, "");

export const dispatchTaskToAgent = async (
  task: Task
): Promise<AgentDispatchResponse> => {
  try {
    const response = await axios.post<ApiResult<AgentDispatchResponse>>(
      `${getAgentBaseUrl()}/api/v1/agent/tasks`,
      task,
      {
        timeout: 10000,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    const payload = response.data;
    if (!payload.errors && payload.data) return payload.data;
    throw new Error(payload.errors?.msg?.[0] || "Failed to dispatch task to agent");
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const detail = error.response?.data?.detail;
      if (typeof detail === "string") throw new Error(detail);
      throw new Error(error.message || "Failed to dispatch task to agent");
    }
    throw error;
  }
};
