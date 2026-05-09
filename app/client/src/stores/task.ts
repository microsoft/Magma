import { Store } from "@tanstack/store";
import type { Action, TaskStatus } from "../types/task";

export interface StepLog {
  step: number;
  action: Action;
  status: TaskStatus;
  message: string;
  timestamp: string;
}

interface TaskStore {
  activeTaskId: string | null;
  isRunning: boolean;
  stepLogs: StepLog[];
  finalStatus: TaskStatus | null;
}

export const taskStore = new Store<TaskStore>({
  activeTaskId: null,
  isRunning: false,
  stepLogs: [],
  finalStatus: null,
});
