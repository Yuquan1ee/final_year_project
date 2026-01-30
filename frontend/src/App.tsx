import { useState } from 'react'
import HomeTab from './components/HomeTab'
import InpaintingTab from './components/InpaintingTab'
import StyleTransferTab from './components/StyleTransferTab'
import RestorationTab from './components/RestorationTab'

// Define the tab types
type Tab = 'home' | 'inpainting' | 'style-transfer' | 'restoration'

function App() {
  // State to track which tab is active
  const [activeTab, setActiveTab] = useState<Tab>('home')

  // Tab configuration - easy to modify
  const tabs: { id: Tab; label: string }[] = [
    { id: 'home', label: 'Home' },
    { id: 'inpainting', label: 'Inpainting' },
    { id: 'style-transfer', label: 'Style Transfer' },
    { id: 'restoration', label: 'Restoration' },
  ]

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-slate-700 p-6">
        <h1 className="text-3xl font-bold text-center">
          DiffusionDesk
        </h1>
        <p className="text-slate-400 text-center mt-2">
          Intelligent image editing powered by diffusion models
        </p>
      </header>

      {/* Tab Navigation */}
      <nav className="border-b border-slate-700">
        <div className="flex gap-1 p-2 max-w-4xl mx-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="p-6 max-w-4xl mx-auto">
        {activeTab === 'home' && <HomeTab />}
        {activeTab === 'inpainting' && <InpaintingTab />}
        {activeTab === 'style-transfer' && <StyleTransferTab />}
        {activeTab === 'restoration' && <RestorationTab />}
      </main>
    </div>
  )
}

export default App
