import { coreApi } from "./core-api";
import type { Task, TaskStatus, TaskProcessRequest, TaskProcessResponse } from "../types/task";

export const createTask = async (prompt: string): Promise<Task> => {
  const response = await coreApi.post<Task>("/api/v1/tasks", { prompt });
  if (response.success && response.data) return response.data;
  throw new Error(response.errors?.msg?.[0] || "Failed to create task");
};

export const listTasks = async (): Promise<Task[]> => {
  const response = await coreApi.get<Task[]>("/api/v1/tasks");
  if (response.success && response.data) return response.data;
  throw new Error(response.errors?.msg?.[0] || "Failed to list tasks");
};

export const getTask = async (taskId: string): Promise<Task> => {
  const response = await coreApi.get<Task>(`/api/v1/tasks/${taskId}`);
  if (response.success && response.data) return response.data;
  throw new Error(response.errors?.msg?.[0] || "Failed to get task");
};

export const getPendingTask = async (): Promise<Task | null> => {
  const response = await coreApi.get<Task | null>("/api/v1/tasks/pending");
  if (response.success) return response.data ?? null;
  throw new Error(response.errors?.msg?.[0] || "Failed to get pending task");
};

export const claimTask = async (taskId: string): Promise<Task> => {
  const response = await coreApi.post<Task>(`/api/v1/tasks/${taskId}/claim`);
  if (response.success && response.data) return response.data;
  throw new Error(response.errors?.msg?.[0] || "Failed to claim task");
};

export const processScreenshot = async (
  payload: TaskProcessRequest
): Promise<TaskProcessResponse> => {
  const response = await coreApi.post<TaskProcessResponse>("/api/v1/tasks/process", payload);
  if (response.success && response.data) return response.data;
  throw new Error(response.errors?.msg?.[0] || "Failed to process screenshot");
};

export const updateTaskStatus = async (taskId: string, status: TaskStatus): Promise<Task> => {
  const response = await coreApi.patch<Task>(`/api/v1/tasks/${taskId}/status`, { status });
  if (response.success && response.data) return response.data;
  throw new Error(response.errors?.msg?.[0] || "Failed to update task status");
};

export const deleteTask = async (taskId: string): Promise<void> => {
  const response = await coreApi.delete(`/api/v1/tasks/${taskId}`);
  if (!response.success) throw new Error(response.errors?.msg?.[0] || "Failed to delete task");
};
