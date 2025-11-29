import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Spin } from 'antd';
import AppLayout from './components/Common/Layout';

// Lazy load pages
const HomePage = lazy(() => import('./pages/Home'));
const PapersPage = lazy(() => import('./pages/Papers'));
const PaperDetailPage = lazy(() => import('./pages/PaperDetailPage'));
const SearchPage = lazy(() => import('./pages/SearchPage'));
const GraphPage = lazy(() => import('./pages/GraphPage'));
const AnalyticsPage = lazy(() => import('./pages/Analytics'));

function App() {
  return (
    <Router>
      <Suspense fallback={<Spin size="large" />}>
        <AppLayout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/papers" element={<PapersPage />} />
            <Route path="/papers/:id" element={<PaperDetailPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/graph" element={<GraphPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AppLayout>
      </Suspense>
    </Router>
  );
}

export default App;
