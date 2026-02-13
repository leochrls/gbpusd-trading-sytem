import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { Landing } from './pages/Landing';
import { Performance } from './pages/Performance';
import { Strategy } from './pages/Strategy';
import { Contact } from './pages/Contact';
import { PublicLayout } from './layouts/PublicLayout';

function App() {
  return (
    <Router>
      <Routes>
        {/* Public Marketing Pages */}
        <Route element={<PublicLayout />}>
          <Route path="/" element={<Landing />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/strategy" element={<Strategy />} />
          <Route path="/contact" element={<Contact />} />
        </Route>

        {/* Technical Dashboard - Standalone Route */}
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
