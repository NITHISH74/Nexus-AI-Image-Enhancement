<div align="center">

# ⚡ Nexus — AI Image Enhancement Suite

> A high-performance **Computer Vision** project showcasing advanced **AI-powered Image Super-Resolution** using ESRGAN, built with a full-stack production mindset.

[![Made with Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](#)
[![Powered by TensorFlow](https://img.shields.io/badge/TensorFlow-%E2%89%A52.16.1-orange.svg?logo=tensorflow&logoColor=white)](#)
[![Built with Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg?logo=flask&logoColor=black)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

</div>

---

## 🌟 Overview

**Nexus** is a cutting-edge AI application designed to enhance, clarify, and upscale low-resolution images into visually sharp, high-quality outputs. Leveraged by **ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)** through TensorFlow Hub, it integrates advanced backend optimization algorithms to ensure blistering speed, hardware scalability, and zero-trust security.

---

## 🏗️ Project Architecture

```text
.
├── Computer_Vision/
│   └── Image_Enhancement_App/
│       ├── flask_app/
│       │   ├── app.py             # Core logic (ESRGAN Inference, Tiling Algorithm)
│       │   ├── static/            # Glassmorphism UI styling
│       │   └── templates/         # Async Fetch DOM overlay
├── .gitignore
└── README.md
└── requirements.txt
```

---

## 🧠 Core Technology

- 🔬 **Inference Model**: ESRGAN (Super-Resolution GAN)  
- ⚙️ **Machine Learning Engine**: TensorFlow & TensorFlow Hub  
- 🌐 **Backend Server**: Python, Flask, Waitress WSGI  
- 🎨 **Frontend Stack**: HTML5, Vanilla JavaScript, CSS3 Glassmorphism  

---

## ✨ Key Features

### 🚀 Intelligent Processing
- **Dynamic Tensor Tiling Algorithm**
  - Safely tessellates exceedingly large resolutions into smaller matrix tiles.
  - Prevents **Out-of-Memory (OOM)** exceptions on standard hardware.
  - Features algorithmic recomposition with zero edge or overlapping visual artifacts.

### ⚡ Blistering Performance Optimization
- **MD5 Hashed LRU Cache**
  - Features an `OrderedDict` LRU memory mapping.
  - Bypasses neural network recomputation for repeat images.
  - Achieves near **`O(1)` theoretical response time** upon instant cache-hits.

### 🔒 Secure Image Handling
- **Volatile In-Memory Streaming**
  - Employs `io.BytesIO` binary streams to mutate images directly in the RAM sequence.
  - Zero payload persistence to physical disk (`/uploads`).
  - Completely mitigates storage bloat and **Directory Traversal** security vulnerabilities.

### 🎨 Premium UI Experience
- Ethereal **Deep-Space Glassmorphism** design.
- Javascript event-listener **Drag-and-Drop** upload bounds.
- Seamless, page-refresh-free SVG loading indicators.
- Interactive, draggable **Before vs After HD comparison slider**.
- Dedicated "Download HD Image" extraction gateway.

---

## 🖥️ Demo Flow

1. **Upload**: Drag & drop an imperfect, low-res image.
2. **Process**: Waitress handles blocking threads while TF infers pixel construction via GANs.
3. **Compare**: Manually trace over the image with the slider trackpad to analyze micro-enhancements.
4. **Extract**: Download the highly upscaled `.jpg` directly to your local file system.

---

## ⚙️ Setup & Installation

### 📌 Prerequisites

- Python **3.8+**  
- TensorFlow **≥ 2.16.1**  

> [!WARNING]
> Due to deprecations in modern Python configurations, TensorFlow Hub expressly requires an older setuptools branch mapping to resolve `pkg_resources`.

### 🚀 Installation Steps

```bash
# 1. Navigate to the local backend application directory
cd Computer_Vision/Image_Enhancement_App/flask_app

# 2. Patch the Setup Tools bug
pip install "setuptools<70"

# 3. Secure isolated dependencies
pip install -r requirements.txt

# 4. Boot the Waitress WSGI Server
python app.py
```

---

## 🌐 Access the Application

Once the terminal outputs `Model loaded successfully`, open your web browser of choice and securely navigate to:

> **[http://localhost:5000](http://localhost:5000)**

---

## 📈 Next-Gen Benchmarks

* ⚡ Highly Accelerated inference thresholds with Hash-caching.
* 🧩 Completely Memory-safe regardless of client payload size.
* 🔄 Exceptionally robust against malicious user requests.

---

## 🤝 Contributing

We welcome advanced PRs! To contribute mathematically or to the UI:

1. `Fork` the repository.
2. Create a specialized branch (`git checkout -b feature/TileOverlapping`).
3. Commit optimizations (`git commit -m "feat: added laplacian filtering"`).
4. `Push` to your branch and open a Pull Request.

---

## 📜 License

This full-stack mechanism represents open-source code and remains available strictly under the **MIT License**.

---

<div align="center">
<b>Built explicitly with a focus on real-world stability, combining production AI performance with pristine UX semantics.</b>
</div>
