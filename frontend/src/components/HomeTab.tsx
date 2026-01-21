/**
 * HomeTab Component
 *
 * Purpose:
 * - Welcome page and introduction to the application
 * - Displays FYP project information
 * - Explains each feature (Inpainting, Style Transfer, Restoration)
 * - Provides background on diffusion models
 */

function HomeTab() {
  const features = [
    {
      title: 'Inpainting',
      description:
        'Remove unwanted objects or fill regions intelligently. Upload an image and mask to let the AI reconstruct the masked areas seamlessly.',
      icon: '🎨',
    },
    {
      title: 'Style Transfer',
      description:
        'Transform your images into different artistic styles like anime, oil painting, watercolor, and more using AI-powered style transfer.',
      icon: '🖼️',
    },
    {
      title: 'Restoration',
      description:
        'Restore old or damaged photos, enhance image quality, remove artifacts, and breathe new life into your memories.',
      icon: '✨',
    },
  ]

  const papers = [
    'Denoising Diffusion Probabilistic Models (DDPM)',
    'Denoising Diffusion Implicit Models (DDIM)',
    'Latent Diffusion Models (Stable Diffusion)',
    'ControlNet',
    'LoRA (Low-Rank Adaptation)',
  ]

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section className="text-center py-8">
        <h2 className="text-3xl font-bold mb-4">
          Diffusion Models for Intelligent Image Editing
        </h2>
        <p className="text-slate-400 max-w-2xl mx-auto">
          This project leverages state-of-the-art diffusion models for image editing,
          inpainting, and style transfer. The AI enables context-aware modifications
          such as object removal, background replacement, and artistic style transformation
          while preserving realism.
        </p>
      </section>

      {/* Project Info */}
      <section className="bg-slate-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Project Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-400">Project #:</span>{' '}
            <span className="text-white">CCDS25-0278</span>
          </div>
          <div>
            <span className="text-slate-400">Student:</span>{' '}
            <span className="text-white">Lee Yu Quan</span>
          </div>
          <div>
            <span className="text-slate-400">Supervisor:</span>{' '}
            <span className="text-white">Prof Zhang Hanwang</span>
          </div>
          <div>
            <span className="text-slate-400">Lab:</span>{' '}
            <span className="text-white">Multimedia and Interacting Computing Lab (MICL)</span>
          </div>
          <div>
            <span className="text-slate-400">Institution:</span>{' '}
            <span className="text-white">Nanyang Technological University</span>
          </div>
          <div>
            <span className="text-slate-400">Academic Year:</span>{' '}
            <span className="text-white">AY 2025/2026</span>
          </div>
        </div>
      </section>

      {/* Features */}
      <section>
        <h3 className="text-xl font-semibold mb-4">Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="bg-slate-800 rounded-lg p-6 hover:bg-slate-750 transition-colors"
            >
              <div className="text-3xl mb-3">{feature.icon}</div>
              <h4 className="text-lg font-semibold mb-2">{feature.title}</h4>
              <p className="text-slate-400 text-sm">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="bg-slate-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">How Diffusion Models Work</h3>
        <p className="text-slate-400 mb-4">
          Diffusion models are a class of generative AI that learn to create images by
          reversing a gradual noising process. During training, the model learns to
          denoise images step by step. During generation, it starts from pure noise
          and iteratively refines it into a coherent image guided by text prompts or
          other conditions.
        </p>
        <div className="flex flex-wrap gap-2">
          {papers.map((paper) => (
            <span
              key={paper}
              className="bg-slate-700 text-slate-300 px-3 py-1 rounded-full text-sm"
            >
              {paper}
            </span>
          ))}
        </div>
      </section>

      {/* Tech Stack */}
      <section>
        <h3 className="text-xl font-semibold mb-4">Technology Stack</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-800 rounded-lg p-4">
            <h4 className="font-medium mb-2">Frontend</h4>
            <p className="text-slate-400 text-sm">React, TypeScript, Vite, Tailwind CSS</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-4">
            <h4 className="font-medium mb-2">Backend</h4>
            <p className="text-slate-400 text-sm">FastAPI, HuggingFace Inference API</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-4">
            <h4 className="font-medium mb-2">AI Models</h4>
            <p className="text-slate-400 text-sm">Stable Diffusion, ControlNet, IP-Adapter</p>
          </div>
          <div className="bg-slate-800 rounded-lg p-4">
            <h4 className="font-medium mb-2">Experiments</h4>
            <p className="text-slate-400 text-sm">PyTorch, Diffusers, NTU HPC Cluster</p>
          </div>
        </div>
      </section>
    </div>
  )
}

export default HomeTab
