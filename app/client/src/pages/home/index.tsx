import type { FC } from "react";
import { useEffect, useRef, useState } from "react";
import {
  Button,
  Card,
  Input,
  Space,
  Table,
  Typography,
  Alert,
  Timeline,
  Spin,
  notification,
} from "antd";
import {
  PlayCircleOutlined,
  StopOutlined,
  DeleteOutlined,
  LoadingOutlined,
} from "@ant-design/icons";
import styled from "styled-components";
import { ContentLayout } from "../../components/layout/content-layout";
import { StatusBadge } from "../../components/common/status-badge";
import { useTaskRunner } from "../../hooks/use-task-runner";
import { useTasksQuery } from "../../services/task.query";
import { useDeleteTaskMutation } from "../../services/task.mutation";
import type { Task, TaskStatus } from "../../types/task";

const TERMINAL_TASK_FEEDBACK: Record<
  Extract<TaskStatus, "done" | "failed" | "cancelled">,
  {
    alertType: "success" | "error" | "warning";
    bannerLabel: string;
    description: string;
    notificationTitle: string;
  }
> = {
  done: {
    alertType: "success",
    bannerLabel: "completed",
    description: "The automation finished successfully.",
    notificationTitle: "Task completed",
  },
  failed: {
    alertType: "error",
    bannerLabel: "failed",
    description: "The automation could not finish successfully.",
    notificationTitle: "Task failed",
  },
  cancelled: {
    alertType: "warning",
    bannerLabel: "cancelled",
    description: "The automation was stopped before finishing.",
    notificationTitle: "Task cancelled",
  },
};

const getDesktopNotification = () => {
  if (typeof window === "undefined" || !("Notification" in window)) return null;
  return window.Notification;
};

const requestDesktopNotificationPermission = async () => {
  const NotificationConstructor = getDesktopNotification();
  if (!NotificationConstructor || NotificationConstructor.permission !== "default") return;

  try {
    await NotificationConstructor.requestPermission();
  } catch {
    // Keep the in-app notification as the fallback.
  }
};

const showDesktopNotification = (title: string, body: string, tag: string) => {
  const NotificationConstructor = getDesktopNotification();
  if (!NotificationConstructor || NotificationConstructor.permission !== "granted") return;

  try {
    const desktopNotification = new NotificationConstructor(title, {
      body,
      tag,
    });

    desktopNotification.onclick = () => {
      window.focus();
      desktopNotification.close();
    };
  } catch {
    // Keep the in-app notification as the fallback.
  }
};

const ActionLogCard = styled(Card)`
  .ant-card-body {
    max-height: 400px;
    overflow-y: auto;
  }
`;

export const HomePage: FC = () => {
  const [prompt, setPrompt] = useState("");
  const [notificationApi, notificationContext] = notification.useNotification();
  const alertedTerminalStateRef = useRef<string | null>(null);
  const { isRunning, stepLogs, activeTaskId, finalStatus, startTask, stopTask } =
    useTaskRunner();
  const { data: tasks = [], isLoading: tasksLoading } = useTasksQuery();
  const deleteMutation = useDeleteTaskMutation();
  const terminalFeedback =
    finalStatus === "done" || finalStatus === "failed" || finalStatus === "cancelled"
      ? TERMINAL_TASK_FEEDBACK[finalStatus]
      : null;

  useEffect(() => {
    if (!terminalFeedback || !finalStatus) return;

    const alertKey = `${activeTaskId ?? "current"}:${finalStatus}`;
    if (alertedTerminalStateRef.current === alertKey) return;
    alertedTerminalStateRef.current = alertKey;

    notificationApi[terminalFeedback.alertType]({
      key: alertKey,
      message: terminalFeedback.notificationTitle,
      description: `${terminalFeedback.description} ${stepLogs.length} step(s) executed.`,
      placement: "topRight",
    });

    showDesktopNotification(
      terminalFeedback.notificationTitle,
      `${terminalFeedback.description} ${stepLogs.length} step(s) executed.`,
      alertKey
    );
  }, [activeTaskId, finalStatus, notificationApi, stepLogs.length, terminalFeedback]);

  const handleStart = async () => {
    await requestDesktopNotificationPermission();
    startTask(prompt);
  };

  const columns = [
    {
      title: "Prompt",
      dataIndex: "prompt",
      key: "prompt",
      ellipsis: true,
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (status: TaskStatus) => <StatusBadge status={status} />,
    },
    {
      title: "Steps",
      dataIndex: "current_step",
      key: "current_step",
      width: 80,
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      render: (val: string) => new Date(val).toLocaleString(),
    },
    {
      title: "",
      key: "actions",
      width: 60,
      render: (_: unknown, record: Task) => (
        <Button
          type="text"
          danger
          icon={<DeleteOutlined />}
          onClick={() => deleteMutation.mutate(record.id)}
          size="small"
        />
      ),
    },
  ];

  return (
    <ContentLayout
      title="UI Automation"
      subtitle="Enter a prompt. This client claims its task, sends it to the local agent, and tracks PyAutoGUI execution."
    >
      {notificationContext}

      {/* Prompt input */}
      <Card className="mb-4">
        <Space.Compact style={{ width: "100%" }}>
          <Input
            placeholder='e.g. "Click the X button to close the dialog"'
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onPressEnter={() => !isRunning && handleStart()}
            disabled={isRunning}
            size="large"
          />
          {!isRunning ? (
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleStart}
              disabled={!prompt.trim()}
              size="large"
            >
              Start Task
            </Button>
          ) : (
            <Button
              danger
              icon={<StopOutlined />}
              onClick={stopTask}
              size="large"
            >
              Stop
            </Button>
          )}
        </Space.Compact>
      </Card>

      {/* Running indicator */}
      {isRunning && (
        <Alert
          type="info"
          showIcon
          icon={<Spin indicator={<LoadingOutlined spin />} size="small" />}
          message="Task in progress"
          description={
            <>
              Task <Typography.Text code>{activeTaskId}</Typography.Text> is
              waiting for this client to claim it and send it to the local
              agent. Make sure the agent API is running.
            </>
          }
          className="mb-4"
        />
      )}

      {/* Completion banner */}
      {!isRunning && terminalFeedback && (
        <Alert
          type={terminalFeedback.alertType}
          message={`Task ${terminalFeedback.bannerLabel} — ${stepLogs.length} step(s) executed`}
          description={terminalFeedback.description}
          className="mb-4"
          closable
        />
      )}

      {/* Action log */}
      {stepLogs.length > 0 && (
        <ActionLogCard title="Action Log" className="mb-4" size="small">
          <Timeline
            items={stepLogs.map((log) => ({
              color:
                log.status === "done"
                  ? "green"
                  : log.status === "cancelled"
                    ? "red"
                    : "blue",
              children: (
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Typography.Text strong>Step {log.step}</Typography.Text>
                    <StatusBadge status={log.status} />
                    <Typography.Text type="secondary" className="text-xs">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </Typography.Text>
                  </div>
                  <div>
                    <Typography.Text code>
                      {log.action.action_type}
                    </Typography.Text>
                    {" — "}
                    {log.action.description}
                  </div>
                  {log.action.parameters &&
                    Object.keys(log.action.parameters).length > 0 && (
                      <Typography.Text type="secondary" className="text-xs">
                        {JSON.stringify(log.action.parameters)}
                      </Typography.Text>
                    )}
                </div>
              ),
            }))}
          />
        </ActionLogCard>
      )}

      {/* Task history */}
      <Card title="Task History" size="small">
        <Table
          dataSource={tasks}
          columns={columns}
          rowKey="id"
          loading={tasksLoading}
          pagination={{ pageSize: 10 }}
          size="small"
          expandable={{
            expandedRowRender: (record: Task) => (
              <div style={{ padding: "8px 0" }}>
                {record.actions_history && record.actions_history.length > 0 ? (
                  <Timeline
                    items={record.actions_history.map((action, idx) => ({
                      color: "blue",
                      children: (
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <Typography.Text strong>
                              Step {idx}
                            </Typography.Text>
                          </div>
                          <div>
                            <Typography.Text code>
                              {action.action_type}
                            </Typography.Text>
                            {" — "}
                            {action.description}
                          </div>
                          {action.parameters &&
                            Object.keys(action.parameters).length > 0 && (
                              <Typography.Text
                                type="secondary"
                                className="text-xs"
                              >
                                {JSON.stringify(action.parameters)}
                              </Typography.Text>
                            )}
                        </div>
                      ),
                    }))}
                  />
                ) : (
                  <Typography.Text type="secondary">
                    No actions recorded yet
                  </Typography.Text>
                )}
              </div>
            ),
            rowExpandable: (record: Task) =>
              record.actions_history && record.actions_history.length > 0,
          }}
        />
      </Card>
    </ContentLayout>
  );
};
