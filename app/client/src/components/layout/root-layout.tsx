import type { FC, ReactNode } from "react";
import { Layout } from "antd";
import { RobotOutlined } from "@ant-design/icons";
import styled from "styled-components";

const { Header } = Layout;

const StyledHeader = styled(Header)`
  display: flex;
  align-items: center;
  gap: 12px;
  background: #001529;
  padding: 0 24px;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: #fff;
  font-size: 18px;
  font-weight: 600;
`;

interface RootLayoutProps {
  children: ReactNode;
}

export const RootLayout: FC<RootLayoutProps> = ({ children }) => {
  return (
    <Layout style={{ minHeight: "100vh" }}>
      <StyledHeader>
        <Logo>
          <RobotOutlined />
          <span>UI Automation Admin</span>
        </Logo>
      </StyledHeader>
      <Layout.Content>{children}</Layout.Content>
    </Layout>
  );
};
