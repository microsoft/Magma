import { useCallback, useEffect, useRef } from "react";
import { useStore } from "@tanstack/react-store";
import { taskStore } from "../stores/task";
import { claimTask, createTask, getTask, updateTaskStatus } from "../services/task";
import { dispatchTaskToAgent } from "../services/agent";
import { useQueryClient } from "@tanstack/react-query";
import { TASK_QUERY_KEYS } from "../services/task.query";
import type { StepLog } from "../stores/task";
import type { Task } from "../types/task";

const POLL_INTERVAL = 2000;

export const useTaskRunner = () => {
  const progressPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const dispatchPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const dispatchedTaskIdsRef = useRef<Set<string>>(new Set());
  const lastStepRef = useRef(0);
  const queryClient = useQueryClient();

  const store = useStore(taskStore);

  const stopProgressPolling = useCallback(() => {
    if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
  }, []);

  const stopDispatchPolling = useCallback(() => {
    if (dispatchPollRef.current) {
      clearInterval(dispatchPollRef.current);
      dispatchPollRef.current = null;
    }
  }, []);

  const stopPolling = useCallback(() => {
    stopProgressPolling();
    stopDispatchPolling();
  }, [stopProgressPolling, stopDispatchPolling]);

  const pollTaskProgress = useCallback(
    async (taskId: string) => {
      try {
        const task: Task = await getTask(taskId);

        // Build new step logs from actions_history that we haven't seen yet
        const newActions = task.actions_history.slice(lastStepRef.current);
        if (newActions.length > 0) {
          const newLogs: StepLog[] = newActions.map((action, i) => ({
            step: lastStepRef.current + i + 1,
            action,
            status: task.status,
            message: `Step ${lastStepRef.current + i + 1} executed`,
            timestamp: new Date().toISOString(),
          }));

          lastStepRef.current = task.actions_history.length;

          taskStore.setState((s) => ({
            ...s,
            stepLogs: [...s.stepLogs, ...newLogs],
          }));
        }

        // Update status
        if (task.status === "done" || task.status === "failed" || task.status === "cancelled") {
          stopPolling();
          dispatchedTaskIdsRef.current.delete(taskId);
          taskStore.setState((s) => ({
            ...s,
            isRunning: false,
            finalStatus: task.status,
          }));
          queryClient.invalidateQueries({ queryKey: TASK_QUERY_KEYS.all });
        }
      } catch {
        // Silently retry on next interval
      }
    },
    [stopPolling, queryClient]
  );

  const pollCreatedTaskForDispatch = useCallback(
    async (taskId: string) => {
      try {
        const task = await getTask(taskId);
        if (task.status !== "pending" || dispatchedTaskIdsRef.current.has(task.id)) return;

        dispatchedTaskIdsRef.current.add(task.id);

        try {
          const claimedTask = await claimTask(task.id);
          await dispatchTaskToAgent(claimedTask);
          stopDispatchPolling();
          queryClient.invalidateQueries({ queryKey: TASK_QUERY_KEYS.all });
        } catch {
          dispatchedTaskIdsRef.current.delete(task.id);
          try {
            await updateTaskStatus(task.id, "pending");
          } catch {
            // Retry this task's claim/dispatch cycle on the next interval.
          }
        }
      } catch {
        // Silently retry on next interval
      }
    },
    [stopDispatchPolling, queryClient]
  );

  const startTask = useCallback(
    async (prompt: string) => {
      if (!prompt.trim()) return;

      // Create task on backend (status: pending)
      const task = await createTask(prompt);

      lastStepRef.current = 0;
      dispatchedTaskIdsRef.current.clear();
      taskStore.setState((s) => ({
        ...s,
        activeTaskId: task.id,
        isRunning: true,
        stepLogs: [],
        finalStatus: null,
      }));

      stopPolling();
      pollCreatedTaskForDispatch(task.id);
      pollTaskProgress(task.id);
      dispatchPollRef.current = setInterval(() => {
        pollCreatedTaskForDispatch(task.id);
      }, POLL_INTERVAL);
      progressPollRef.current = setInterval(() => {
        pollTaskProgress(task.id);
      }, POLL_INTERVAL);
    },
    [pollCreatedTaskForDispatch, pollTaskProgress, stopPolling]
  );

  const stopTask = useCallback(async () => {
    stopPolling();
    const taskId = taskStore.state.activeTaskId;
    if (taskId) dispatchedTaskIdsRef.current.delete(taskId);
    if (taskId) {
      try {
        await updateTaskStatus(taskId, "cancelled");
      } catch {
        // Server may be unreachable, still stop locally
      }
    }
    taskStore.setState((s) => ({ ...s, isRunning: false, finalStatus: "cancelled" }));
    queryClient.invalidateQueries({ queryKey: TASK_QUERY_KEYS.all });
  }, [stopPolling, queryClient]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return {
    ...store,
    startTask,
    stopTask,
  };
};
