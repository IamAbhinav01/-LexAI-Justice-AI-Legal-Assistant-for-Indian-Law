# ⚖️ LexAI Justice - Advanced Legal AI Assistant

A comprehensive Streamlit-based legal AI assistant with advanced features including voice input/output, dark/light themes, and intelligent legal guidance powered by Indian law documents.

## 🚀 Performance Features

### ⚡ Model Caching & Persistence
- **Smart Model Caching**: Models are cached using Streamlit's `@st.cache_resource` for instant reloads
- **Persistent Sessions**: Models stay loaded across app refreshes and browser sessions
- **Fast Startup**: Subsequent app loads are 10-50x faster than initial load
- **Memory Efficient**: Optimized model loading with proper resource management

### 🎯 Performance Optimizations
- **Concurrent Request Handling**: Up to 5 simultaneous requests with semaphore control
- **Response Caching**: Frequently asked questions cached for instant responses
- **Resource Cleanup**: Automatic garbage collection and memory management
- **Connection Pooling**: Optimized database connections with CassIO

## 🎨 Features

### 🎤 Voice Features
- **Speech-to-Text**: Real-time voice input using Whisper
- **Text-to-Speech**: Audio responses using TTS
- **Voice Recording**: In-app microphone recording
- **Audio Playback**: Integrated audio player for responses

### 🎨 UI/UX
- **Dark/Light Themes**: Toggle between themes
- **Professional Design**: Modern gradient backgrounds
- **Responsive Layout**: Optimized for different screen sizes
- **Progress Indicators**: Real-time loading feedback

### 🤖 AI Capabilities
- **Advanced RAG**: Retrieval-Augmented Generation with legal documents
- **Context Awareness**: Maintains conversation history
- **Legal Expertise**: Trained on Indian legal documents
- **Structured Responses**: Professional formatting with sections and tables

### 📊 Monitoring
- **Performance Metrics**: Real-time response times and memory usage
- **Model Status**: Live status of all AI models
- **Cache Management**: Manual cache clearing and model reloading
- **Resource Monitoring**: Memory and CPU usage tracking

## ScreenShots
![Screenshot 2025-07-06 213109](https://github.com/user-attachments/assets/7117a45c-0dee-40f0-bbe1-fa82190c9d27)
## Demo
- visit : https://www.linkedin.com/in/abhinav-sunil-870184279/ 
## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd LegalAI

# Install dependencies
pip install -r requirements.txt

# Configure AstraDB (optional)
cp config.json.example config.json
# Edit config.json with your AstraDB credentials

# Run the application
python run.py
# or
streamlit run main.py
```

## ⚙️ Configuration

### AstraDB Setup (Optional)
1. Create an AstraDB account at https://astra.datastax.com/
2. Create a new database
3. Get your token, database ID, and keyspace
4. Update `config.json`:

```json
{
    "astra_token": "your_astra_token",
    "database_id": "your_database_id", 
    "keyspace": "your_keyspace",
    "table_name": "legal_documents",
    "nvidia_api_key": "your_nvidia_api_key"
}
```

### Demo Mode
If AstraDB is not configured, the app runs in demo mode with:
- Sample legal responses
- All UI features available
- Voice input/output (if models are available)

## 🎯 Usage

### Basic Usage
1. **Text Input**: Type your legal question in the text area
2. **Voice Input**: Click the microphone button and speak
3. **Get Response**: Receive detailed legal guidance with citations
4. **Listen**: Click the speaker button to hear the response

### Advanced Features
- **Theme Switching**: Toggle between dark/light themes
- **Chat History**: View and manage conversation history
- **Performance Controls**: Monitor and optimize app performance
- **Model Management**: Reload models or check their status

### Performance Tips
- **First Load**: Initial startup may take 30-60 seconds (models loading)
- **Subsequent Loads**: Should be 2-5 seconds (cached models)
- **Cache Management**: Use "Clear Cache" for fresh responses
- **Resource Cleanup**: Use "Cleanup Resources" if app becomes slow

## 🔧 Troubleshooting

### Common Issues

#### App Hangs on Startup
```bash
# Clear Streamlit cache
streamlit cache clear

# Kill existing processes
taskkill /f /im streamlit.exe  # Windows
pkill -f streamlit  # Linux/Mac

# Restart with clean cache
streamlit run main.py --server.port 8501
```

#### Model Loading Errors
```bash
# Check Python version (should be 3.8+)
python --version

# Reinstall dependencies
pip uninstall -r requirements.txt
pip install -r requirements.txt

# Test model loading
python test_caching.py
```

#### Port Already in Use
```bash
# Use different port
streamlit run main.py --server.port 8502

# Or kill existing process
netstat -ano | findstr :8501  # Windows
lsof -i :8501  # Linux/Mac
```

### Performance Optimization
- **GPU Usage**: Ensure CUDA is available for faster processing
- **Memory**: Close other applications to free up RAM
- **Cache**: Use "Reload Models" only when necessary
- **Network**: Stable internet for model downloads

## 📁 Project Structure

```
LegalAI/
├── main.py                 # Main application file
├── config.json             # Configuration file
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Hugging Face** for the AI models
- **AstraDB** for vector database capabilities
- **Indian Legal Community** for the legal documents

---

**⚖️ LexAI Justice** - Making legal guidance accessible to everyone! 🇮🇳 
