# -*- coding: utf-8 -*-
"""
LexAI Justice - Complete Legal AI Assistant
Single comprehensive application with stunning UI and full backend integration
"""
config = {
    "astra_token": os.getenv("ASTRA_TOKEN"),
    "database_id": os.getenv("DATABASE_ID"),
    "keyspace": os.getenv("KEYSPACE"),
    "table_name": os.getenv("TABLE_NAME"),
    "nvidia_api_key": os.getenv("NVIDIA_API_KEY")
}
import streamlit as st
import time
import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
import threading
import queue
import numpy as np
import hashlib
from functools import lru_cache
import gc
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging for performance monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance monitoring
class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.avg_response_time = 0
        self.memory_usage = []
        self.cpu_usage = []
    
    def log_request(self, response_time: float):
        """Log request performance metrics"""
        self.request_count += 1
        self.avg_response_time = (self.avg_response_time * (self.request_count - 1) + response_time) / self.request_count
        
        # Monitor system resources
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent()
        
        self.memory_usage.append(memory_percent)
        self.cpu_usage.append(cpu_percent)
        
        # Keep only last 100 measurements
        if len(self.memory_usage) > 100:
            self.memory_usage.pop(0)
            self.cpu_usage.pop(0)
        
        logger.info(f"Request {self.request_count}: {response_time:.2f}s, Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_requests": self.request_count,
            "avg_response_time": self.avg_response_time,
            "avg_memory_usage": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_usage": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "uptime": time.time() - self.start_time
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Response cache for improved performance
class ResponseCache:
    """Simple in-memory cache for responses"""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        if key in self.cache:
            # Check if cache entry is still valid
            if time.time() - self.access_times[key] < self.ttl:
                self.access_times[key] = time.time()
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key: str, value: str):
        """Set cached response"""
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
        self.access_times.clear()

# Global response cache
response_cache = ResponseCache()

# Try to import backend components (with fallbacks)
BACKEND_AVAILABLE = False
WHISPER_AVAILABLE = False
TTS_AVAILABLE = False
CASSANDRA_AVAILABLE = False
VECTOR_DB_AVAILABLE = False
NVIDIA_AVAILABLE = False

try:
    import sounddevice as sd
    from scipy.io.wavfile import write, read
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores.cassandra import Cassandra
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    BACKEND_AVAILABLE = True
    VECTOR_DB_AVAILABLE = True
except ImportError as e:
    print(f"Basic backend components not available: {e}")

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError as e:
    print(f"Whisper not available: {e}")

try:
    import TTS.api as TTS
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"TTS not available: {e}")

try:
    import cassio
    CASSANDRA_AVAILABLE = True
except ImportError as e:
    print(f"Cassandra not available: {e}")

try:
    from langchain_nvidia import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError as e:
    print(f"NVIDIA not available: {e}")

# Page configuration
st.set_page_config(
    page_title="LexAI Justice - AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
class Config:
    CONFIG_PATH = "config.json"
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    AUDIO_CHUNK_SIZE = 1024
    RECORDING_DURATION = 6
    WHISPER_MODEL_SIZE = "base"  # Smaller model for faster performance
    TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # Back to original model with better error handling
    EMBEDDING_MODEL = "BAAI/bge-base-en"  # Use the original model that matches dimensions
    LLM_MODEL = "mistralai/mixtral-8x7b-instruct-v0.1"
    DATA_DIR = "data"
    
    # Performance optimization settings
    MAX_CACHE_SIZE = 100  # Maximum number of cached responses
    CACHE_TTL = 3600  # Cache time-to-live in seconds
    MAX_CONCURRENT_REQUESTS = 3  # Limit concurrent processing
    BATCH_SIZE = 5  # Batch size for processing
    ENABLE_CACHING = True  # Enable response caching
    ENABLE_BATCHING = True  # Enable request batching

# Sample legal responses (fallback)
SAMPLE_RESPONSES = {
    "theft": """**Theft under Indian Penal Code (IPC)**

**Section 378 - Theft**
Whoever, intending to take dishonestly any moveable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft.

**Punishment (Section 379):**
- Imprisonment up to 3 years
- Fine, or both

**Key Elements:**
1. **Dishonest intention** - Must intend to cause wrongful gain or wrongful loss
2. **Moveable property** - Cannot steal immovable property
3. **Without consent** - Must be taken without owner's permission
4. **Physical movement** - Property must be moved to constitute theft

*This is a simplified explanation. For specific legal advice, consult a qualified lawyer.*""",

    "murder": """**Murder under Indian Penal Code (IPC)**

**Section 300 - Murder**
Except in the cases hereinafter excepted, culpable homicide is murder, if the act by which the death is caused is done with the intention of causing death, or—

**Punishment (Section 302):**
- Death penalty, or
- Imprisonment for life, and
- Fine

**Key Elements:**
1. **Intention to cause death** - Must have mens rea
2. **Culpable homicide** - Unlawful killing of a human being
3. **Causation** - Act must cause death
4. **No legal justification** - Not in self-defense or other exceptions

*This is a simplified explanation. For specific legal advice, consult a qualified lawyer.*""",

    "assault": """**Assault and Battery under Indian Penal Code (IPC)**

**Section 351 - Assault**
Whoever makes any gesture, or any preparation intending or knowing it to be likely that such gesture or preparation will cause any person present to apprehend that he who makes that gesture or preparation is about to use criminal force to that person, is said to commit an assault.

**Section 352 - Punishment for Assault**
- Imprisonment up to 3 months
- Fine up to ₹500, or both

**Key Differences:**
- **Assault**: Threat of force (no actual contact)
- **Battery**: Actual use of force
- **Hurt**: Bodily pain, disease, or infirmity
- **Grievous Hurt**: Serious injuries (fractures, permanent damage, etc.)

*This is a simplified explanation. For specific legal advice, consult a qualified lawyer.*"""
}

class BackendService:
    """Backend service for AI, database, and voice processing with performance optimizations"""
    
    def __init__(self):
        self.config = Config()
        self.session_id = str(uuid.uuid4())
        
        # Use cached models instead of loading them every time
        self.whisper_model = load_whisper_model()
        self.tts_model = load_tts_model()
        self.embeddings_model = load_embeddings_model()
        self.llm_model = load_llm_model()
        
        self.vector_store = None
        self.rag_chain = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.chat_history_store = {}
        
        # Performance optimizations
        self.request_semaphore = threading.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_CONCURRENT_REQUESTS)
        self.model_cache = {}
        self.last_gc_time = time.time()
        
        # Initialize components that don't need model loading
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all backend components with performance optimizations"""
        try:
            # Models are already loaded via caching, just check status
            if self.whisper_model:
                st.success("✅ Whisper model loaded successfully (cached)")
            else:
                st.warning("⚠️ Whisper not available - using demo mode")
            
            if self.tts_model:
                st.success("✅ TTS model loaded successfully (cached)")
            else:
                st.warning("⚠️ TTS not available - using demo mode")
            
            # Initialize AstraDB Vector Store
            if CASSANDRA_AVAILABLE and VECTOR_DB_AVAILABLE and self.embeddings_model:
                self._initialize_astra_vector_store()
            else:
                st.warning("⚠️ Vector database not available - using demo mode")
            
            # Force garbage collection after initialization
            gc.collect()
            
        except Exception as e:
            st.warning(f"⚠️ Some components failed to initialize: {e}")
    
    def _initialize_astra_vector_store(self):
        """Initialize AstraDB Vector Store with CassIO and performance optimizations"""
        try:
            # Load config with caching
            config = load_config()
            if not config:
                st.warning("📁 Config file not found or invalid")
                return
            
            # Initialize CassIO with connection pooling
            cassio.init(
                token=config['astra_token'],
                database_id=config['database_id'],
                keyspace=config['keyspace']
            )
            
            # Set NVIDIA API key if available
            if 'nvidia_api_key' in config and config['nvidia_api_key'] != "your_nvidia_api_key_here":
                os.environ["NVIDIA_API_KEY"] = config['nvidia_api_key']
            
            # Use cached embeddings model
            if not self.embeddings_model:
                st.warning("⚠️ Embeddings model not available")
                return
            
            # Initialize AstraDB Vector Store
            self.vector_store = Cassandra(
                embedding=self.embeddings_model,
                table_name=config['table_name'],
                session=None,
                keyspace=config['keyspace'],
            )
            
            # Use cached LLM model
            if not self.llm_model:
                st.warning("⚠️ LLM model not available - using fallback")
                return
            
            # Create retriever with optimized settings
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.7  # Only return relevant results
                }
            )
            
            # Create system prompt
            system_prompt = """You are LexAI 🇮🇳, a helpful legal assistant trained in Indian laws. 
Respond empathetically, clearly, and in a professional GPT-style format with structured sections, emojis where appropriate, and markdown formatting.

Your answer **must include** the following:
- 👋 Warm greeting with empathy
- 🧑‍⚖️ Legal sections with bold IPC/POCSO names and brief punishment summaries
- 🏫 Actionable steps the user can take
- 🫂 Emotional support guidance
- 📌 Summary table (Markdown format)
- ✅ Disclaimer at the end

Always cite relevant laws like IPC, CrPC, POCSO, JJ Act by name and section number.

Respond to:
{context}

Chat History:
{chat_history}

Question:
{input}
"""
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # Create chains with performance optimizations
            combine_docs_chain = create_stuff_documents_chain(
                self.llm_model, 
                prompt,
                document_variable_name="context"
            )
            self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            # Add message history
            self.rag_chain = RunnableWithMessageHistory(
                self.rag_chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            
            st.success("✅ AstraDB Vector Store and RAG chain initialized successfully")
            
        except Exception as e:
            st.warning(f"⚠️ AstraDB initialization failed: {e}")
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session with memory management"""
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = ChatMessageHistory()
        
        # Limit chat history size to prevent memory issues
        history = self.chat_history_store[session_id]
        if len(history.messages) > 50:  # Keep only last 50 messages
            history.messages = history.messages[-50:]
        
        return history
    
    @lru_cache(maxsize=100)
    def _get_cached_transcription(self, audio_hash: str) -> str:
        """Get cached transcription result"""
        return response_cache.get(audio_hash)
    
    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using Whisper with caching and performance optimizations"""
        start_time = time.time()
        
        try:
            if self.whisper_model and WHISPER_AVAILABLE:
                # Create hash of audio data for caching
                audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
                
                # Check cache first
                cached_result = self._get_cached_transcription(audio_hash)
                if cached_result:
                    performance_monitor.log_request(time.time() - start_time)
                    return cached_result
                
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    write(temp_file.name, self.config.AUDIO_SAMPLE_RATE, audio_data)
                    
                    # Transcribe with timeout
                    try:
                        segments, _ = self.whisper_model.transcribe(
                            temp_file.name,
                            beam_size=1,  # Faster decoding
                            best_of=1     # Reduce computation
                        )
                        transcription = " ".join([segment.text for segment in segments])
                        
                        # Cache the result
                        response_cache.set(audio_hash, transcription.strip())
                        
                        # Clean up
                        os.unlink(temp_file.name)
                        
                        performance_monitor.log_request(time.time() - start_time)
                        return transcription.strip()
                        
                    except Exception as e:
                        logger.error(f"Transcription timeout or error: {e}")
                        return " What is the punishment for theft under IPC?"
                        
            else:
                return "What is the punishment for theft under IPC?"
                
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
            return ""
    
    def generate_response(self, user_input: str) -> str:
        """Generate AI response using RAG chain with caching and performance optimizations"""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.config.ENABLE_CACHING:
                cache_key = hashlib.md5(user_input.encode()).hexdigest()
                cached_response = response_cache.get(cache_key)
                if cached_response:
                    performance_monitor.log_request(time.time() - start_time)
                    return cached_response
            
            # Use semaphore to limit concurrent requests
            with self.request_semaphore:
                if self.rag_chain and VECTOR_DB_AVAILABLE and self.vector_store:
                    try:
                        # Use RAG chain for response
                        config = {"configurable": {"session_id": self.session_id}}
                        response = self.rag_chain.invoke(
                            {"input": user_input},
                            config=config
                        )
                        result = response["answer"]
                        
                        # Cache the result
                        if self.config.ENABLE_CACHING:
                            response_cache.set(cache_key, result)
                        
                        performance_monitor.log_request(time.time() - start_time)
                        return result
                    except Exception as e:
                        st.warning(f"⚠️ RAG chain failed, using fallback: {e}")
                        # Fall through to fallback responses
                else:
                    st.info("ℹ️ Running in demo mode - using sample responses")
                
                # Fallback to sample responses
                user_input_lower = user_input.lower()
                if "theft" in user_input_lower:
                    result = SAMPLE_RESPONSES["theft"]
                elif "murder" in user_input_lower:
                    result = SAMPLE_RESPONSES["murder"]
                elif "assault" in user_input_lower or "battery" in user_input_lower:
                    result = SAMPLE_RESPONSES["assault"]
                else:
                    result = """**Legal Guidance**

I understand you're asking about legal matters. This is a demonstration version of LexAI Justice.

**Available Topics:**
- **Theft** - Ask about theft laws and punishments
- **Murder** - Information about murder and homicide
- **Assault** - Details about assault and battery

**For comprehensive legal advice:**
- Consult a qualified lawyer
- Visit official legal websites
- Contact legal aid services

*This is a demo version. The full version includes advanced AI capabilities, voice interaction, and comprehensive legal database access.*"""
                
                performance_monitor.log_request(time.time() - start_time)
                return result
                    
        except Exception as e:
            st.error(f"❌ Response generation failed: {e}")
            performance_monitor.log_request(time.time() - start_time)
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech with caching and performance optimizations"""
        start_time = time.time()
        
        try:
            if self.tts_model and TTS_AVAILABLE:
                # Validate input text
                if not text or len(text.strip()) < 3:
                    st.warning("⚠️ Text too short for TTS. Minimum 3 characters required.")
                    return b""
                
                # Check cache first
                text_hash = hashlib.md5(text.encode()).hexdigest()
                cached_audio = response_cache.get(f"tts_{text_hash}")
                if cached_audio:
                    performance_monitor.log_request(time.time() - start_time)
                    return cached_audio.encode() if isinstance(cached_audio, str) else cached_audio
                
                # Generate speech
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    try:
                        # Check if model supports speakers and languages
                        if hasattr(self.tts_model, 'speakers') and self.tts_model.speakers:
                            # Multi-speaker model
                            if hasattr(self.tts_model, 'languages') and self.tts_model.languages:
                                # Multi-lingual model
                                self.tts_model.tts_to_file(
                                    text=text, 
                                    file_path=temp_file.name, 
                                    language="en", 
                                    speaker="male-en-2"
                                )
                            else:
                                # Single-language, multi-speaker model
                                self.tts_model.tts_to_file(
                                    text=text, 
                                    file_path=temp_file.name, 
                                    speaker="male-en-2"
                                )
                        else:
                            # Single-speaker model (like tacotron2-DDC)
                            if hasattr(self.tts_model, 'languages') and self.tts_model.languages:
                                # Multi-lingual, single-speaker model
                                self.tts_model.tts_to_file(
                                    text=text, 
                                    file_path=temp_file.name, 
                                    language="en"
                                )
                            else:
                                # Single-language, single-speaker model
                                self.tts_model.tts_to_file(
                                    text=text, 
                                    file_path=temp_file.name
                                )
                        
                        # Read the generated audio
                        with open(temp_file.name, 'rb') as f:
                            audio_data = f.read()
                        
                        # Cache the result
                        response_cache.set(f"tts_{text_hash}", audio_data)
                        
                        # Clean up
                        os.unlink(temp_file.name)
                        
                        performance_monitor.log_request(time.time() - start_time)
                        return audio_data
                        
                    except Exception as tts_error:
                        st.warning(f"⚠️ TTS generation failed: {tts_error}")
                        # Clean up temp file if it exists
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                        
                        # Try fallback TTS with shorter text
                        return self._fallback_tts(text)
            else:
                return b""  # Return empty bytes for demo
                
        except Exception as e:
            st.error(f"❌ Text-to-speech failed: {e}")
            return b""
    
    def _fallback_tts(self, text: str) -> bytes:
        """Fallback TTS method for when main TTS fails"""
        try:
            # Use a shorter, simpler text for fallback
            fallback_text = text[:100] if len(text) > 100 else text
            if len(fallback_text) < 10:
                fallback_text = "Thank you for using LexAI Justice."
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Try with minimal parameters
                self.tts_model.tts_to_file(
                    text=fallback_text, 
                    file_path=temp_file.name
                )
                
                # Read the generated audio
                with open(temp_file.name, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up
                os.unlink(temp_file.name)
                
                return audio_data
                
        except Exception as e:
            st.warning(f"⚠️ Fallback TTS also failed: {e}")
            return b""
    
    def cleanup_resources(self):
        """Clean up resources and perform garbage collection"""
        try:
            # Clear caches
            response_cache.clear()
            
            # Clear model cache
            self.model_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"❌ Resource cleanup failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = performance_monitor.get_stats()
        stats.update({
            "cache_size": len(response_cache.cache),
            "chat_sessions": len(self.chat_history_store),
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(),
            "gpu_memory_used": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        })
        return stats
    
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
    
    def stop_recording(self) -> np.ndarray:
        """Stop audio recording and return audio data"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        # Get recorded audio from queue
        audio_data = np.array([])
        try:
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get_nowait()
                audio_data = np.append(audio_data, chunk)
        except queue.Empty:
            pass
        
        return audio_data
    
    def _record_audio(self):
        """Record audio in background thread"""
        try:
            with sd.InputStream(
                samplerate=self.config.AUDIO_SAMPLE_RATE,
                channels=self.config.AUDIO_CHANNELS,
                dtype=np.float32,
                blocksize=self.config.AUDIO_CHUNK_SIZE
            ) as stream:
                while self.is_recording:
                    audio_chunk, _ = stream.read(self.config.AUDIO_CHUNK_SIZE)
                    self.audio_queue.put(audio_chunk.flatten())
                    
        except Exception as e:
            st.error(f"❌ Audio recording failed: {e}")

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'backend' not in st.session_state:
        st.session_state.backend = BackendService()
    
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"
    
    if 'language' not in st.session_state:
        st.session_state.language = "English"
    
    if 'tone' not in st.session_state:
        st.session_state.tone = "Professional"

def render_header():
    """Render the main header with theme toggle"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🌙" if st.session_state.theme == "light" else "☀️", key="theme_toggle"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="margin: 0; color: #1f77b4;">⚖️ LexAI Justice</h1>
            <p style="margin: 0; color: #666; font-size: 1.1em;">Your AI Legal Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: right;">
            <p style="margin: 0; color: #666; font-size: 0.9em;">Powered by AI</p>
        </div>
        """, unsafe_allow_html=True)

def render_chat_bubble(message: str, sender: str, timestamp: datetime):
    """Render a chat bubble with proper styling"""
    if sender == "user":
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px; 
                        max-width: 70%; word-wrap: break-word; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        color: white; padding: 12px 16px; border-radius: 18px 18px 18px 4px; 
                        max-width: 70%; word-wrap: break-word; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_chat_history():
    """Render the chat history"""
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            render_chat_bubble(
                message["content"], 
                message["role"], 
                message.get("timestamp", datetime.now())
            )

def handle_mic_click():
    """Handle microphone button click"""
    if not st.session_state.is_recording:
        st.session_state.is_recording = True
        st.session_state.backend.start_recording()
        st.rerun()
    else:
        st.session_state.is_recording = False
        audio_data = st.session_state.backend.stop_recording()
        
        if len(audio_data) > 0:
            # Transcribe audio
            transcription = st.session_state.backend.transcribe_audio(audio_data)
            
            if transcription:
                # Add user message
                st.session_state.messages.append({
                    "role": "user", 
                    "content": transcription,
                    "timestamp": datetime.now()
                })
                
                # Generate response
                response = st.session_state.backend.generate_response(transcription)
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": datetime.now()
                })
                
                st.rerun()

def handle_submit(user_input: str):
    """Handle text input submission"""
    if user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Generate response
        response = st.session_state.backend.generate_response(user_input)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now()
        })
        
        st.rerun()

def render_input_section():
    """Render the input section with voice and text options"""
    st.markdown("---")
    
    # Voice input section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        mic_button = st.button(
            "🎙️" if not st.session_state.is_recording else "⏹️",
            key="mic_button",
            help="Click to start/stop voice recording"
        )
        if mic_button:
            handle_mic_click()
    
    with col2:
        if st.session_state.is_recording:
            st.markdown("""
            <div style="text-align: center; color: #ff4444; font-weight: bold;">
                🎙️ Recording... Click again to stop
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            if TTS_AVAILABLE:
                tts_button = st.button("🔊", key="tts_button", help="Listen to the last response")
                if tts_button:
                    last_response = st.session_state.messages[-1]["content"]
                    audio_data = st.session_state.backend.text_to_speech(last_response)
                    if audio_data:
                        st.audio(audio_data, format="audio/wav")
                    else:
                        st.warning("⚠️ Audio generation failed. Please try again.")
            else:
                st.button("🔊", key="tts_button_disabled", help="TTS not available", disabled=True)
    
    # Text input section
    st.markdown("---")
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Ask your legal question:",
                placeholder="e.g., What is the punishment for theft under IPC?",
                height=100,
                key="user_input"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            submit_button = st.form_submit_button("Send", type="primary")
        
        if submit_button and user_input.strip():
            handle_submit(user_input)

def render_sidebar():
    """Render the sidebar with settings and features"""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Theme selection
        theme = st.selectbox(
            "Theme",
            ["dark", "light"],
            index=0 if st.session_state.theme == "dark" else 1
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
        
        # Language selection
        language = st.selectbox(
            "Language",
            ["English", "Hindi", "Gujarati", "Bengali"],
            index=0
        )
        if language != st.session_state.language:
            st.session_state.language = language
        
        # Tone selection
        tone = st.selectbox(
            "Response Tone",
            ["Professional", "Friendly", "Formal", "Casual"],
            index=0
        )
        if tone != st.session_state.tone:
            st.session_state.tone = tone
        
        st.markdown("---")
        
        # Features section
        st.markdown("## 🚀 Features")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Export chat
        if st.button("📥 Export Chat", key="export_chat"):
            if st.session_state.messages:
                chat_text = ""
                for msg in st.session_state.messages:
                    chat_text += f"{msg['role'].title()}: {msg['content']}\n\n"
                
                st.download_button(
                    label="Download Chat",
                    data=chat_text,
                    file_name=f"lexai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        st.markdown("---")
        
        # Status section
        st.markdown("## 📊 Status")
        
        status_items = [
            ("🤖 AI Model", "✅ Active" if VECTOR_DB_AVAILABLE else "❌ Demo Mode"),
            ("🎙️ Voice Input", "✅ Active" if WHISPER_AVAILABLE else "❌ Demo Mode"),
            ("🔊 Voice Output", "✅ Active" if TTS_AVAILABLE else "❌ Demo Mode"),
            ("🗄️ Database", "✅ Connected" if CASSANDRA_AVAILABLE else "❌ Demo Mode"),
        ]
        
        for item, status in status_items:
            st.markdown(f"**{item}:** {status}")

def render_status_panel():
    """Render the status panel at the bottom with performance metrics"""
    st.markdown("---")
    
    # Get performance stats
    try:
        stats = st.session_state.backend.get_performance_stats()
    except:
        stats = {}
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    
    with col2:
        st.metric("Session ID", st.session_state.backend.session_id[:8] + "...")
    
    with col3:
        avg_response_time = stats.get('avg_response_time', 0)
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    with col4:
        cache_size = stats.get('cache_size', 0)
        st.metric("Cache Size", cache_size)
    
    with col5:
        memory_mb = stats.get('memory_usage_mb', 0)
        st.metric("Memory Usage", f"{memory_mb:.0f}MB")
    
    # Performance details in expander
    with st.expander("📊 Performance Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Requests", stats.get('total_requests', 0))
            st.metric("CPU Usage", f"{stats.get('avg_cpu_usage', 0):.1f}%")
        
        with col2:
            st.metric("Uptime", f"{stats.get('uptime', 0)/60:.1f}min")
            if torch.cuda.is_available():
                gpu_memory = stats.get('gpu_memory_used', 0)
                st.metric("GPU Memory", f"{gpu_memory:.0f}MB")
        
        with col3:
            st.metric("Chat Sessions", stats.get('chat_sessions', 0))
            st.metric("Theme", st.session_state.theme.title())
        
        # Performance optimization controls
        st.markdown("### ⚡ Performance Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear Cache", key="clear_cache"):
                response_cache.clear()
                st.success("Cache cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Cleanup Resources", key="cleanup_resources"):
                st.session_state.backend.cleanup_resources()
                st.success("Resources cleaned up!")
                st.rerun()

def main():
    """Main application function"""
    try:
        # Show startup progress
        with st.spinner("🚀 Initializing LexAI Justice..."):
            # Initialize session state
            init_session_state()
            
            # Show model loading progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load models with progress indication
            status_text.text("Loading Whisper model...")
            progress_bar.progress(25)
            
            status_text.text("Loading TTS model...")
            progress_bar.progress(50)
            
            status_text.text("Loading embeddings model...")
            progress_bar.progress(75)
            
            status_text.text("Loading LLM model...")
            progress_bar.progress(90)
            
            status_text.text("Initializing components...")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Apply theme
        if st.session_state.theme == "dark":
            st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Render header
        render_header()
        
        # Show demo mode warning if needed
        if not VECTOR_DB_AVAILABLE or not CASSANDRA_AVAILABLE:
            st.warning("""
            ⚠️ **Demo Mode Active**
            
            Some components are not available. The app is running in demo mode with sample responses.
            
            **Available Features:**
            - ✅ Text input and responses
            - ✅ Chat interface
            - ✅ Theme switching
            - ✅ Performance monitoring
            
            **Demo Topics:**
            - Ask about "theft" laws
            - Ask about "murder" laws  
            - Ask about "assault" laws
            """)
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Chat area
            chat_container = st.container()
            
            with chat_container:
                render_chat_history()
                
                # Auto-scroll to bottom
                if st.session_state.messages:
                    st.markdown("""
                    <script>
                    window.scrollTo(0, document.body.scrollHeight);
                    </script>
                    """, unsafe_allow_html=True)
            
            # Input section
            render_input_section()
            
            # Status panel
            render_status_panel()
        
        with col2:
            # Sidebar
            render_sidebar()
            
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.info("""
        **Troubleshooting:**
        1. Check if all dependencies are installed
        2. Ensure config.json exists with valid credentials
        3. Try running in demo mode
        """)
        
        # Show basic interface even if there's an error
        st.markdown("## ⚖️ LexAI Justice - Demo Mode")
        st.text_area("Ask a question:", placeholder="e.g., What is theft?")
        if st.button("Send"):
            st.info("Demo response: This is a demonstration of LexAI Justice. In full mode, you would get detailed legal guidance here.")

# Global model cache for persistence across reloads
@st.cache_resource
def load_whisper_model():
    """Load Whisper model with caching"""
    if WHISPER_AVAILABLE:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel(
                "base", 
                device=device, 
                compute_type=compute_type,
                cpu_threads=4,
                num_workers=1
            )
            return model
        except Exception as e:
            st.warning(f"⚠️ Whisper model loading failed: {e}")
            return None
    return None

@st.cache_resource
def load_tts_model():
    """Load TTS model with caching"""
    if TTS_AVAILABLE:
        try:
            model = TTS.TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False, 
                gpu=torch.cuda.is_available()
            )
            return model
        except Exception as e:
            st.warning(f"⚠️ TTS model loading failed: {e}")
            return None
    return None

@st.cache_resource
def load_embeddings_model():
    """Load embeddings model with caching"""
    if VECTOR_DB_AVAILABLE:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return embeddings
        except Exception as e:
            st.warning(f"⚠️ Embeddings model loading failed: {e}")
            return None
    return None

@st.cache_resource
def load_llm_model():
    """Load LLM model with caching"""
    if NVIDIA_AVAILABLE:
        try:
            model = ChatNVIDIA(
                model="mistralai/mixtral-8x7b-instruct-v0.1",
                temperature=0.1,
                max_tokens=500
            )
            return model
        except Exception as e:
            st.warning(f"⚠️ LLM model loading failed: {e}")
            return None
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_config():
    """Load configuration with caching"""
    try:
        if os.path.exists("config.json"):
            with open("config.json", 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"⚠️ Config loading failed: {e}")
    return None

if __name__ == "__main__":
    main() 
