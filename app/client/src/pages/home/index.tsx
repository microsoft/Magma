import type { FC } from "react";
import { useState } from "react";
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

const ActionLogCard = styled(Card)`
  .ant-card-body {
    max-height: 400px;
    overflow-y: auto;
  }
`;

export const HomePage: FC = () => {
  const [prompt, setPrompt] = useState("");
  const { isRunning, stepLogs, activeTaskId, finalStatus, startTask, stopTask } =
    useTaskRunner();
  const { data: tasks = [], isLoading: tasksLoading } = useTasksQuery();
  const deleteMutation = useDeleteTaskMutation();

  const handleStart = () => {
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
      subtitle="Enter a prompt. The agent running in the background will pick it up, take screenshots, and execute PyAutoGUI actions automatically."
    >
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
              waiting for the agent to pick it up and execute. Make sure the
              agent is running and pointed at the backend.
            </>
          }
          className="mb-4"
        />
      )}

      {/* Completion banner */}
      {!isRunning && finalStatus && stepLogs.length > 0 && (
        <Alert
          type={finalStatus === "done" ? "success" : "warning"}
          message={`Task ${finalStatus === "done" ? "completed" : "stopped"} — ${stepLogs.length} step(s) executed`}
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
