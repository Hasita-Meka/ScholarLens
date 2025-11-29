// src/components/Common/Footer.tsx
import React from 'react';
import { Layout } from 'antd';
import { GithubOutlined, LinkedinOutlined, TwitterOutlined } from '@ant-design/icons';

const { Footer: AntFooter } = Layout;

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <AntFooter className="footer" style={{ textAlign: 'center', marginTop: '50px' }}>
      <div className="footer-content">
        <div className="footer-links">
          <a href="/about">About</a>
          <span className="divider">|</span>
          <a href="/privacy">Privacy Policy</a>
          <span className="divider">|</span>
          <a href="/contact">Contact</a>
        </div>
        <div className="footer-social" style={{ marginTop: '10px' }}>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer">
            <GithubOutlined />
          </a>
          <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" style={{ marginLeft: '15px' }}>
            <LinkedinOutlined />
          </a>
          <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" style={{ marginLeft: '15px' }}>
            <TwitterOutlined />
          </a>
        </div>
        <p style={{ marginTop: '15px', marginBottom: '0' }}>
          © {currentYear} ScholarLens. All rights reserved.
        </p>
      </div>
    </AntFooter>
  );
};

export default Footer;