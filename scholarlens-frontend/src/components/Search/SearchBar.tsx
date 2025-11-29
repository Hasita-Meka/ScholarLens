import React, { useState } from 'react';
import { Input, Button, Select, Space } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const SearchBar: React.FC = () => {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState<'semantic' | 'keyword'>('semantic');
  const navigate = useNavigate();

  const handleSearch = () => {
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query)}&type=${searchType}`);
    }
  };

  return (
    <Space>
      <Input
        placeholder="Search papers..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onPressEnter={handleSearch}
        style={{ width: '300px' }}
      />
      <Select
        value={searchType}
        onChange={setSearchType}
        style={{ width: '120px' }}
      >
        <Select.Option value="semantic">Semantic</Select.Option>
        <Select.Option value="keyword">Keyword</Select.Option>
      </Select>
      <Button type="primary" icon={<SearchOutlined />} onClick={handleSearch}>
        Search
      </Button>
    </Space>
  );
};

export default SearchBar;
