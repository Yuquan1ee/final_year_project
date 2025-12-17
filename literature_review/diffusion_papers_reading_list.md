# Essential Papers for Diffusion Models & Image Generation

A comprehensive reading list for understanding modern diffusion models, from foundational theory to state-of-the-art applications like Stable Diffusion, ControlNet, and Sora.

---

## 📚 Core Diffusion Model Papers

### 1. Denoising Diffusion Probabilistic Models (DDPM)
**Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel  
**Year:** 2020  
**Conference:** NeurIPS 2020  

**Key Contribution:** The breakthrough paper that made diffusion models practical for high-quality image generation.

- **arXiv:** https://arxiv.org/abs/2006.11239
- **PDF:** https://arxiv.org/pdf/2006.11239.pdf
- **GitHub:** https://github.com/hojonathanho/diffusion
- **Project Page:** https://hojonathanho.github.io/diffusion/

**Why Read:** This is THE foundational paper. Introduces the training and sampling procedures that all modern diffusion models build upon.

---

### 2. Denoising Diffusion Implicit Models (DDIM)
**Authors:** Jiaming Song, Chenlin Meng, Stefano Ermon  
**Year:** 2020  
**Conference:** ICLR 2021  

**Key Contribution:** Made diffusion models 10-50× faster by enabling deterministic sampling with fewer steps.

- **arXiv:** https://arxiv.org/abs/2010.02502
- **PDF:** https://arxiv.org/pdf/2010.02502.pdf
- **GitHub:** https://github.com/ermongroup/ddim

**Why Read:** Essential for understanding how to speed up diffusion models for practical applications.

---

### 3. Score-Based Generative Modeling through Stochastic Differential Equations
**Authors:** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole  
**Year:** 2021  
**Conference:** ICLR 2021 (Outstanding Paper Award)  

**Key Contribution:** Provides unified theoretical framework connecting diffusion models and score-based models through SDEs.

- **arXiv:** https://arxiv.org/abs/2011.13456
- **PDF:** https://arxiv.org/pdf/2011.13456.pdf
- **GitHub:** https://github.com/yang-song/score_sde
- **Blog Post:** https://yang-song.net/blog/2021/score/

**Why Read:** Best for understanding the theoretical foundations and continuous-time perspective of diffusion.

---

## 🎨 Latent Diffusion & Stable Diffusion

### 4. High-Resolution Image Synthesis with Latent Diffusion Models
**Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer  
**Year:** 2022  
**Conference:** CVPR 2022 (Oral)  

**Key Contribution:** THE Stable Diffusion paper. Runs diffusion in compressed latent space instead of pixel space, making high-resolution generation feasible.

- **arXiv:** https://arxiv.org/abs/2112.10752
- **PDF:** https://arxiv.org/pdf/2112.10752.pdf
- **GitHub (LDM):** https://github.com/CompVis/latent-diffusion
- **GitHub (Stable Diffusion):** https://github.com/CompVis/stable-diffusion
- **Project Page:** https://ommer-lab.com/research/latent-diffusion-models/

**Why Read:** This is the architecture behind Stable Diffusion. Essential for understanding modern text-to-image models.

---

## 🎮 Conditional Generation & Control

### 5. Classifier-Free Diffusion Guidance
**Authors:** Jonathan Ho, Tim Salimans  
**Year:** 2022  
**Venue:** NeurIPS 2021 Workshop (extended arXiv version 2022)  

**Key Contribution:** Enables strong conditioning without a separate classifier by training conditional and unconditional models jointly.

- **arXiv:** https://arxiv.org/abs/2207.12598
- **PDF:** https://arxiv.org/pdf/2207.12598.pdf

**Why Read:** Critical technique used in all modern text-to-image models for guidance strength control.

---

### 6. Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)
**Authors:** Lvmin Zhang, Anyi Rao, Maneesh Agrawala  
**Year:** 2023  
**Conference:** ICCV 2023  

**Key Contribution:** Adds precise spatial control (edges, depth, pose, etc.) to pretrained diffusion models using zero convolutions.

- **arXiv:** https://arxiv.org/abs/2302.05543
- **PDF:** https://arxiv.org/pdf/2302.05543.pdf
- **GitHub:** https://github.com/lllyasviel/ControlNet
- **Hugging Face:** https://huggingface.co/lllyasviel/ControlNet

**Why Read:** Revolutionary for controllable image generation. Shows how to add spatial conditioning to existing models.

---

### 7. Prompt-to-Prompt Image Editing with Cross-Attention Control
**Authors:** Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, Daniel Cohen-Or  
**Year:** 2022  

**Key Contribution:** Understanding how text conditioning works through cross-attention mechanisms.

- **arXiv:** https://arxiv.org/abs/2208.01626
- **PDF:** https://arxiv.org/pdf/2208.01626.pdf
- **GitHub:** https://github.com/google/prompt-to-prompt

**Why Read:** Important for understanding attention mechanisms in text-to-image generation.

---

## 🔤 Text Encoders & Cross-Modal Understanding

### 8. Learning Transferable Visual Models From Natural Language Supervision (CLIP)
**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever  
**Year:** 2021  
**Conference:** ICML 2021  

**Key Contribution:** Learns joint image-text representations through contrastive learning. Used as text encoder in most text-to-image models.

- **arXiv:** https://arxiv.org/abs/2103.00020
- **PDF:** https://arxiv.org/pdf/2103.00020.pdf
- **GitHub:** https://github.com/openai/CLIP
- **Blog Post:** https://openai.com/blog/clip/

**Why Read:** Understanding CLIP is essential since it's the text encoder in Stable Diffusion and many other models.

---

## 🤖 Transformer-Based Diffusion

### 9. Scalable Diffusion Models with Transformers (DiT)
**Authors:** William Peebles, Saining Xie  
**Year:** 2023  
**Conference:** ICCV 2023 (Oral)  

**Key Contribution:** Replaces U-Net with Vision Transformers in diffusion models, showing better scalability.

- **arXiv:** https://arxiv.org/abs/2212.09748
- **PDF:** https://arxiv.org/pdf/2212.09748.pdf
- **GitHub:** https://github.com/facebookresearch/DiT
- **Project Page:** https://www.wpeebles.com/DiT

**Why Read:** Represents the future direction of diffusion models. Architecture likely related to Sora.

---

## 🎬 Video Diffusion (Sora-Related)

### 10. Video Diffusion Models
**Authors:** Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet  
**Year:** 2022  

**Key Contribution:** First major work extending diffusion models to video generation.

- **arXiv:** https://arxiv.org/abs/2204.03458
- **PDF:** https://arxiv.org/pdf/2204.03458.pdf
- **Project Page:** https://video-diffusion.github.io/

**Why Read:** Foundational work for understanding video generation with diffusion.

---

### 11. Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models
**Authors:** Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, Karsten Kreis  
**Year:** 2023  

**Key Contribution:** Extends Latent Diffusion to video by adding temporal layers. Can turn Stable Diffusion into video model.

- **arXiv:** https://arxiv.org/abs/2304.08818
- **PDF:** https://arxiv.org/pdf/2304.08818.pdf
- **Project Page:** https://research.nvidia.com/labs/toronto-ai/VideoLDM/

**Why Read:** Shows how to adapt image diffusion models for video generation.

---

### 12. Imagen Video: High Definition Video Generation with Diffusion Models
**Authors:** Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans  
**Year:** 2022  

**Key Contribution:** Google's approach to high-quality video generation with cascaded diffusion models.

- **arXiv:** https://arxiv.org/abs/2210.02303
- **PDF:** https://arxiv.org/pdf/2210.02303.pdf
- **Project Page:** https://imagen.research.google/video/

**Why Read:** Alternative approach to video diffusion using cascaded architecture.

---

## 📐 Earlier Foundation Work (For Context)

### 13. Generative Adversarial Networks (GANs)
**Authors:** Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
**Year:** 2014  
**Conference:** NeurIPS 2014  

- **arXiv:** https://arxiv.org/abs/1406.2661
- **PDF:** https://arxiv.org/pdf/1406.2661.pdf

**Why Read:** Understanding the predecessor to diffusion models provides useful context.

---

### 14. Auto-Encoding Variational Bayes (VAE)
**Authors:** Diederik P. Kingma, Max Welling  
**Year:** 2013  
**Conference:** ICLR 2014  

- **arXiv:** https://arxiv.org/abs/1312.6114
- **PDF:** https://arxiv.org/pdf/1312.6114.pdf

**Why Read:** VAEs are used in Stable Diffusion for encoding/decoding the latent space.

---

## 📖 Recommended Reading Order

### Week 1-2: Fundamentals
1. **DDPM (2020)** - Core diffusion mechanism
2. **DDIM (2020)** - Faster sampling methods  
3. **Score-Based SDE (2021)** - Theoretical framework

**Goal:** Understand the basic training and sampling procedures of diffusion models.

---

### Week 3: Practical Implementation
4. **Latent Diffusion Models (2022)** - The Stable Diffusion paper
5. **Classifier-Free Guidance (2022)** - Text conditioning technique
6. **CLIP (2021)** - Text encoder (skim if needed)

**Goal:** Understand how modern text-to-image models work in practice.

---

### Week 4: Advanced Control & Architecture
7. **ControlNet (2023)** - Spatial control mechanisms
8. **DiT (2023)** - Transformer-based architecture

**Goal:** Learn about state-of-the-art control and architectural improvements.

---

### Week 5: Video Generation (Optional)
9. **Video Diffusion Models (2022)** - Basic video extension
10. **Video LDMs (2023)** - Latent video models
11. **Imagen Video (2022)** - Cascaded approach

**Goal:** If your project involves video, understand temporal consistency and generation.

---

## 🛠️ Useful Resources

### Tutorials & Blogs
- **Lil'Log Blog:** https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- **Yang Song's Blog:** https://yang-song.net/blog/2021/score/
- **Hugging Face Diffusers:** https://huggingface.co/docs/diffusers/

### Code Repositories
- **Hugging Face Diffusers:** https://github.com/huggingface/diffusers
- **Stable Diffusion WebUI:** https://github.com/AUTOMATIC1111/stable-diffusion-webui
- **ComfyUI:** https://github.com/comfyanonymous/ComfyUI

### Open Source Models
- **Stable Diffusion Models:** https://huggingface.co/stabilityai
- **ControlNet Models:** https://huggingface.co/lllyasviel
- **DiT Checkpoints:** https://github.com/facebookresearch/DiT#pre-trained-dit-models

---

## 📝 Notes

- All papers listed are freely available on arXiv
- GitHub repositories contain official implementations where available
- This list focuses on image/video generation - diffusion models are also used for audio, 3D, etc.
- For your final year project, focus on papers 1-8 as core reading
- Papers 9-12 are optional depending on whether you're working with video

---

## 🎯 Quick Reference by Topic

**Understanding Core Mechanism:** Papers #1, #2, #3  
**Building Text-to-Image Models:** Papers #4, #5, #8  
**Adding Control:** Papers #6, #7  
**Video Generation:** Papers #10, #11, #12  
**Transformer Architecture:** Paper #9  

---

**Last Updated:** December 2024  
**Created for:** Final Year Project on Image Generation/Diffusion Models

Good luck with your project! 🚀
