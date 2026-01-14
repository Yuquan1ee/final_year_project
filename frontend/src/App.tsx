import { useState, useEffect } from 'react';
import { Paintbrush, Wand2, Palette, CheckCircle, XCircle } from 'lucide-react';
import { InpaintingTab } from './components/InpaintingTab';
import { EditingTab } from './components/EditingTab';
import { StyleTransferTab } from './components/StyleTransferTab';
import { healthCheck } from './services/api';
import './App.css';

type TabType = 'inpainting' | 'editing' | 'style';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('editing');
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    const checkApi = async () => {
      const isHealthy = await healthCheck();
      setApiStatus(isHealthy ? 'online' : 'offline');
    };
    checkApi();
    // Check every 30 seconds
    const interval = setInterval(checkApi, 30000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'editing' as const, label: 'Image Editing', icon: Wand2 },
    { id: 'inpainting' as const, label: 'Inpainting', icon: Paintbrush },
    { id: 'style' as const, label: 'Style Transfer', icon: Palette },
  ];

  return (
    <div className="app">
      <header className="header">
        <h1>Diffusion Image Editor</h1>
        <p className="subtitle">Intelligent image editing powered by diffusion models</p>
        <div className={`api-status ${apiStatus}`}>
          {apiStatus === 'checking' && <span>Checking API...</span>}
          {apiStatus === 'online' && (
            <>
              <CheckCircle size={14} />
              <span>API Online</span>
            </>
          )}
          {apiStatus === 'offline' && (
            <>
              <XCircle size={14} />
              <span>API Offline - Start the backend server</span>
            </>
          )}
        </div>
      </header>

      <nav className="tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <tab.icon size={18} />
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="main">
        {activeTab === 'inpainting' && <InpaintingTab />}
        {activeTab === 'editing' && <EditingTab />}
        {activeTab === 'style' && <StyleTransferTab />}
      </main>

      <footer className="footer">
        <p>FYP Project: Diffusion Models for Intelligent Image Editing and Inpainting</p>
      </footer>
    </div>
  );
}

export default App;
