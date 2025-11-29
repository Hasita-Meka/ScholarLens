// src/pages/SearchPage.tsx
import React, { useState } from 'react';
import { Layout, Row, Col } from 'antd';
import SearchBar from '@components/Search/SearchBar';
import SearchFilters from '@components/Search/SearchFilters';
import SearchResults from '@components/Search/SearchResults';

const { Content } = Layout;

export const SearchPage: React.FC = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = (query: string, type: string) => {
  setLoading(true);
  try {
    // Fetch results from API
  } finally {
    setLoading(false);
  }
};


  return (
    <Layout>
      <Content style={{ padding: '20px' }}>
        <SearchBar />
        <Row gutter={20} style={{ marginTop: '20px' }}>
          <Col xs={24} sm={24} md={6}>
            <SearchFilters onApply={() => {}} />
          </Col>
          <Col xs={24} sm={24} md={18}>
            <SearchResults results={results} loading={loading} />
          </Col>
        </Row>
      </Content>
    </Layout>
  );
};

export default SearchPage;