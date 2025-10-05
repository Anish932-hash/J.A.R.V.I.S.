# J.A.R.V.I.S. 2.0 - Installation Guide
## Ultra-Advanced AI Personal Assistant for Windows

> **"Sometimes you've got to run before you can walk." - Tony Stark**

This guide will walk you through setting up J.A.R.V.I.S. with all its advanced features enabled.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [API Keys Setup](#api-keys-setup)
6. [Advanced Features Setup](#advanced-features-setup)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.9 or higher (3.10+ recommended)
- **RAM**: 8GB
- **Storage**: 10GB free space
- **CPU**: Quad-core processor
- **Internet**: Broadband connection

### Recommended for Full AI Features
- **RAM**: 16GB+ (32GB for optimal performance)
- **Storage**: 50GB+ SSD (for AI models and data)
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or better)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (for neural network acceleration)
- **CUDA**: CUDA 11.8+ installed

---

## Prerequisites

### 1. Python Installation
```bash
# Download Python 3.10+ from python.org
# Make sure to check "Add Python to PATH" during installation

# Verify installation
python --version
```

### 2. Git Installation
```bash
# Download Git from git-scm.com
# Verify installation
git --version
```

### 3. Visual C++ Build Tools (Required for some packages)
Download and install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### 4. CUDA Toolkit (Optional, for GPU acceleration)
Download from: https://developer.nvidia.com/cuda-downloads
- Required for PyTorch GPU support
- Install CUDA 11.8 or 12.x

### 5. Additional Tools
- **Tesseract OCR**: Download from https://github.com/UB-Mannheim/tesseract/wiki
  - Add to PATH: `C:\Program Files\Tesseract-OCR`
- **FFmpeg**: Download from https://ffmpeg.org/download.html
  - Add to PATH for multimedia processing

---

## Installation Methods

### Method 1: Quick Installation (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd jarvis

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 4. Install core dependencies first
pip install psutil pywin32 pyttsx3 SpeechRecognition pyaudio PyQt6

# 5. Install AI/ML frameworks (this may take a while)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentence-transformers chromadb

# 6. Install remaining dependencies
pip install -r requirements.txt

# 7. Download spaCy language model
python -m spacy download en_core_web_sm

# 8. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 9. Run system diagnostics
python main.py --diagnostics

# 10. Launch J.A.R.V.I.S.
python main.py
```

### Method 2: Docker Installation (Experimental)

```bash
# Build Docker image
docker-compose build

# Run J.A.R.V.I.S.
docker-compose up -d

# View logs
docker-compose logs -f jarvis-core

# Access shell
docker-compose exec jarvis-core bash
```

### Method 3: Step-by-Step Installation

#### Step 1: Environment Setup
```bash
cd jarvis
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

#### Step 2: Install Core System Packages
```bash
pip install psutil>=6.0.0
pip install pywin32>=306
pip install colorama>=0.4.6
```

#### Step 3: Install Voice Interface
```bash
pip install pyttsx3>=2.90
pip install SpeechRecognition>=3.10.0

# PyAudio (may require Microsoft C++ Build Tools)
pip install pyaudio>=0.2.13

# If PyAudio fails, download wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# Then: pip install PyAudio‑0.2.13‑cp310‑cp310‑win_amd64.whl
```

#### Step 4: Install GUI Framework
```bash
pip install PyQt6>=6.7.1
pip install PyQt6-Qt6>=6.7.1
pip install pyqtgraph matplotlib plotly
```

#### Step 5: Install AI/ML Stack
```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version
# pip install torch torchvision torchaudio

# Transformers and NLP
pip install transformers>=4.44.2
pip install sentence-transformers>=3.1.1
pip install spacy>=3.7.6
python -m spacy download en_core_web_sm

# Vector databases
pip install chromadb>=0.5.5
pip install faiss-cpu>=1.8.0

# TensorFlow (optional)
pip install tensorflow>=2.17.0
```

#### Step 6: Install Web & Networking
```bash
pip install requests aiohttp beautifulsoup4
pip install selenium playwright
pip install websockets paho-mqtt
```

#### Step 7: Install Computer Vision
```bash
pip install opencv-python Pillow
pip install pytesseract mss
pip install face-recognition mediapipe
```

#### Step 8: Install Additional Features
```bash
pip install pandas numpy scipy
pip install scikit-learn xgboost
pip install cryptography PyJWT
pip install pytest pytest-asyncio
```

---

## Configuration

### 1. Create Configuration Directory
```bash
mkdir -p config
mkdir -p data
mkdir -p logs
```

### 2. Generate Default Configuration
```bash
# Run JARVIS once to generate default config
python main.py --config config/jarvis.json
```

### 3. Edit Configuration (config/jarvis.json)
```json
{
  "system": {
    "name": "J.A.R.V.I.S.",
    "version": "2.0.0",
    "auto_start": true,
    "enable_voice": true,
    "enable_gui": true,
    "log_level": "INFO"
  },
  "voice": {
    "engine": "windows",
    "voice": "default",
    "rate": 200,
    "volume": 0.8,
    "wake_word": "jarvis"
  },
  "ai": {
    "enable_transformers": true,
    "enable_neural_networks": true,
    "gpu_acceleration": true,
    "model_cache_dir": "data/models"
  },
  "monitoring": {
    "cpu_threshold": 80,
    "memory_threshold": 85,
    "disk_threshold": 90,
    "enable_alerts": true
  },
  "security": {
    "enable_encryption": true,
    "enable_authentication": false
  }
}
```

---

## API Keys Setup

J.A.R.V.I.S. can integrate with 100+ AI services. Set up API keys for the services you want to use.

### 1. Create Environment File
```bash
# Create .env file in project root
touch .env
```

### 2. Add API Keys (.env)
```bash
# OpenAI (GPT-4, DALL-E)
OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Google AI
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-custom-search-engine-id

# Microsoft Azure
AZURE_SPEECH_KEY=your-azure-speech-key
AZURE_SPEECH_REGION=your-region

# Stability AI (Stable Diffusion)
STABILITY_API_KEY=sk-your-stability-key

# ElevenLabs (Voice)
ELEVENLABS_API_KEY=your-elevenlabs-key

# Cohere
COHERE_API_KEY=your-cohere-key

# Search APIs
BING_SEARCH_API_KEY=your-bing-key
SERPAPI_KEY=your-serpapi-key

# Other services
REPLICATE_API_TOKEN=your-replicate-token
HUGGINGFACE_API_KEY=your-huggingface-key
```

### 3. Test API Connections
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); print('Environment loaded')"
```

---

## Advanced Features Setup

### 1. GPU Acceleration Setup

#### Check CUDA Installation
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

#### Install GPU-specific packages
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Vector Database Setup (ChromaDB)

```bash
# ChromaDB is already installed, but you can configure it
# The default database location is: data/memory_db/

# Test ChromaDB
python -c "import chromadb; client = chromadb.Client(); print('ChromaDB working!')"
```

### 3. Voice Recognition Setup

#### Test Microphone
```bash
python test_voice.py
```

#### Configure Voice Settings
Edit `config/jarvis.json`:
```json
{
  "voice": {
    "engine": "windows",  // or "azure" for better quality
    "voice": "default",   // or specific voice name
    "rate": 200,          // Speech rate
    "volume": 0.8,        // Volume (0.0 to 1.0)
    "wake_word": "jarvis",
    "emotion_detection": true,
    "noise_reduction": true
  }
}
```

### 4. Neural Network Models Setup

```bash
# Download pre-trained models
python -c "from transformers import pipeline; pipe = pipeline('text-generation'); print('Models downloaded')"

# Download sentence transformers
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print('Embeddings ready')"
```

### 5. IoT Integration Setup (Optional)

```bash
# Install MQTT broker (Mosquitto)
# Download from: https://mosquitto.org/download/

# Configure MQTT in config/jarvis.json
{
  "iot": {
    "enabled": true,
    "mqtt_broker": "localhost",
    "mqtt_port": 1883,
    "mqtt_username": "",
    "mqtt_password": ""
  }
}
```

### 6. Distributed Computing Setup (Optional)

```bash
# Install Redis
# Download from: https://github.com/microsoftarchive/redis/releases

# Install Celery
pip install celery redis

# Start Redis
redis-server

# Start Celery worker
celery -A jarvis.core.distributed worker --loglevel=info
```

---

## Running J.A.R.V.I.S.

### Basic Launch
```bash
python main.py
```

### Launch with GUI
```bash
python main.py --gui
```

### Launch with Voice Only
```bash
python main.py --voice-only
```

### Launch with Full AI Stack
```bash
python main.py --ai-full
```

### Run Diagnostics
```bash
python main.py --diagnostics --deep-scan
```

### Launch Terminal Interface
```bash
python run_terminal.py
```

---

## Troubleshooting

### PyAudio Installation Issues
```bash
# Download pre-built wheel
# Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# Download appropriate wheel for your Python version
pip install PyAudio‑0.2.13‑cp310‑cp310‑win_amd64.whl
```

### CUDA/GPU Not Detected
```bash
# Verify NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ImportError: DLL load failed
```bash
# Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### Voice Recognition Not Working
```bash
# Check microphone permissions in Windows Settings
# Test microphone
python test_voice.py

# If using Azure Speech, verify API key
python -c "import os; print(os.getenv('AZURE_SPEECH_KEY'))"
```

### ChromaDB Errors
```bash
# Clear ChromaDB data
rm -rf data/memory_db

# Reinstall ChromaDB
pip uninstall chromadb
pip install chromadb>=0.5.5
```

### Memory Issues
```bash
# Reduce batch size in config
# Disable some AI features
# Use CPU instead of GPU for some operations
```

---

## Next Steps

1. **Read the README.md** for feature overview
2. **Check ADVANCED_FEATURES.md** for detailed feature documentation
3. **Explore the GUI** by running `python main.py --gui`
4. **Try voice commands** like "Jarvis, what's the weather?"
5. **Read API documentation** in `docs/api_reference.md`
6. **Create plugins** using the plugin system

---

## Getting Help

- Check logs in `logs/` directory
- Run diagnostics: `python main.py --diagnostics`
- Review configuration: `config/jarvis.json`
- Check GitHub Issues for known problems
- Join our community for support

---

## Performance Tips

1. **Use SSD** for faster model loading
2. **Enable GPU acceleration** for AI features
3. **Allocate more RAM** to Python process
4. **Close unnecessary applications** when running AI models
5. **Use fast internet** for API calls
6. **Cache API responses** to reduce costs
7. **Batch operations** when possible

---

**J.A.R.V.I.S. 2.0** - Built for the future. Designed for today.
