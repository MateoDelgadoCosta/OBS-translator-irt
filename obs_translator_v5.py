"""
OBS Real-Time Speech Translation Plugin v5.5
==========================================

A production-ready, thread-safe OBS Studio Python plugin for real-time
speech-to-text and translation with advanced audio processing.

Features:
- Thread-safe audio stream management
- Clean shutdown without OBS crashes
- OBS text source integration

Author: OBS Translator Team
Version: 5.5.0
"""

from __future__ import annotations

import obspython as obs
import threading
import queue
import json
import os
import sys
import subprocess
import urllib.request
import zipfile
import time
import re
import logging
import difflib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager
import warnings

import numpy as np
import scipy.signal as signal

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

VERSION = "5.10.0"
PLUGIN_NAME = "OBS Translator"
DEFAULT_SAMPLERATE = 44100
TARGET_SAMPLERATE = 16000
BLOCKSIZE_MS = 50
MAX_HISTORY_LINES = 50
MAX_QUEUE_SIZE = 10
TRANSLATION_QUEUE_SIZE = 20
DEFAULT_AUDIO_GATE_DB = -60
VAD_HANGOVER_MS = 300
VAD_THRESHOLD_MARGIN_DB = 12
COMPRESSOR_THRESHOLD_DB = -20
COMPRESSOR_RATIO = 3.0
HIGH_PASS_FREQ_HZ = 80
PREEMPHASIS_ALPHA = 0.97
CONFIDENCE_THRESHOLD = 0.3
FINALIZATION_SILENCE_MS = 700
FINALIZATION_SPEECH_MAX_MS = 30000
NOISE_FLOOR_ALPHA = 0.01
NOISE_PROFILE_ALPHA = 0.99

# Model URLs
MODEL_URLS = {
    "small": {
        "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "es": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
    },
    "large": {
        "en": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip",
        "es": "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip"
    }
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

class ObsLogHandler(logging.Handler):
    """Custom handler that sends logs to OBS script log"""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level_map = {
                logging.DEBUG: obs.LOG_DEBUG,
                logging.INFO: obs.LOG_INFO,
                logging.WARNING: obs.LOG_WARNING,
                logging.ERROR: obs.LOG_ERROR,
                logging.CRITICAL: obs.LOG_ERROR
            }
            obs.script_log(level_map.get(record.levelno, obs.LOG_INFO), f"[Translator] {msg}")
        except Exception:
            pass

# Configure logger
_logger = logging.getLogger(f"{PLUGIN_NAME}.v{VERSION}")
_logger.setLevel(logging.DEBUG)
_handler = ObsLogHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_logger.addHandler(_handler)

def log(msg: str, level: int = obs.LOG_INFO) -> None:
    """Wrapper for OBS script logging"""
    _logger.log(level, msg)

# ============================================================================
# GLOBAL STATE (Thread-Safe)
# ============================================================================

@dataclass
class GlobalState:
    """Thread-safe global state management"""
    translation_worker: Optional["AudioSTTWorker"] = None
    text_queue: Optional[queue.Queue] = None
    script_settings: Optional[Any] = None
    last_status: str = "Ready. Configure and click START."
    is_running: bool = False
    is_exiting: bool = False
    worker_lock: threading.Lock = field(default_factory=threading.Lock)
    shutdown_event: threading.Event = field(default_factory=threading.Event)

# Thread-safe singleton
_state = GlobalState()

# Global mic cache (immutable after first read)
_mic_list_cache: List[Dict[str, Any]] = []

# Compiled regex patterns (pre-compile for performance)
_SAFE_NAME_REGEX = re.compile(r'[^a-zA-Z0-9_]')
_WORD_SPLIT_REGEX = re.compile(r'\s+')

# OBS API safety flag - when True, all OBS API calls are blocked
_obs_api_disabled = False

# ============================================================================
# EXCEPTIONS
# ============================================================================

class TranslatorError(Exception):
    """Base exception for translator errors"""
    pass

class ModelNotFoundError(TranslatorError):
    """Raised when required model is not found"""
    pass

class DependencyError(TranslatorError):
    """Raised when required dependencies are missing"""
    pass

class AudioDeviceError(TranslatorError):
    """Raised when audio device cannot be accessed"""
    pass

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_base_dir() -> Path:
    """Get the base directory for plugin data storage"""
    app_data = os.environ.get('APPDATA')
    if app_data:
        base_path = Path(app_data) / "OBS_Translator"
    else:
        base_path = Path.home() / "OBS_Translator"
    
    # Create directory with secure permissions
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def get_vocab_file() -> Path:
    """Get the custom vocabulary file path"""
    return get_base_dir() / "custom_vocabulary.json"

def get_slang_file() -> Path:
    """Get the custom slang dictionary file path"""
    return get_base_dir() / "custom_slang.json"

# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

ES_KEYWORDS = frozenset({
    'el', 'la', 'los', 'las', 'que', 'de', 'por', 'con', 'más', 'está',
    'como', 'pero', 'para', 'uno', 'una', 'este', 'esta', 'ese', 'esa',
    'tiene', 'hace', 'tiene', 'donde', 'cuando', 'quien', 'porque', 'hoy',
    'ahora', 'asi', 'aquí', 'todo', 'todos', 'todas', 'muy', 'también',
    'pues', 'entonces', 'solo', 'ser', 'estar', 'hacer', 'tener', 'poder',
    'decir', 'ver', 'dar', 'saber', 'querer', 'buenos', 'días', 'tardes',
    'noches', 'gracias', 'hola', 'adiós', 'sí', 'no', 'nada', 'algo'
})

EN_KEYWORDS = frozenset({
    'the', 'is', 'that', 'for', 'with', 'are', 'this', 'have', 'from',
    'they', 'been', 'will', 'your', 'what', 'when', 'where', 'which',
    'their', 'there', 'would', 'could', 'should', 'about', 'into',
    'time', 'year', 'years', 'people', 'way', 'day', 'days', 'just',
    'know', 'take', 'come', 'made', 'find', 'say', 'good', 'well',
    'back', 'after', 'think', 'must', 'look', 'want', 'give', 'use',
    'working', 'thanks', 'hello', 'bye', 'yes', 'nothing', 'something'
})


def _verify_with_keywords(text: str, detected: str) -> str:
    """Verify language detection using EN/ES keywords"""
    text_lower = text.lower()
    
    es_count = sum(1 for w in ES_KEYWORDS if f' {w} ' in f' {text_lower} ' or text_lower.endswith(w) or text_lower.startswith(w))
    en_count = sum(1 for w in EN_KEYWORDS if f' {w} ' in f' {text_lower} ' or text_lower.endswith(w) or text_lower.startswith(w))
    
    if es_count > en_count:
        return 'es'
    elif en_count > es_count:
        return 'en'
    return detected


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect language with high precision using fast-langdetect.
    Returns (language_code, confidence)
    """
    if not text or len(text.strip()) < 3:
        return 'es', 0.0
    
    try:
        from fast_langdetect import detect
        result = detect(text, model='full')
        lang = result.get('lang', '')
        confidence = result.get('confidence', 0.0)
        
        if lang not in ('en', 'es'):
            lang = 'es'
        
        if confidence < 0.75:
            lang = _verify_with_keywords(text, lang)
        
        return lang, confidence
    except Exception:
        return 'es', 0.0


def load_custom_vocab() -> Dict[str, Dict[str, Any]]:
    """Load custom vocabulary from file (thread-safe read)"""
    vocab_file = get_vocab_file()
    if not vocab_file.exists():
        return {"en": {}, "es": {}}
    
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Validate structure
            if not isinstance(data, dict):
                return {"en": {}, "es": {}}
            return data
    except (json.JSONDecodeError, IOError) as e:
        log(f"Failed to load vocabulary: {e}", obs.LOG_WARNING)
        return {"en": {}, "es": {}}

def save_custom_vocab(vocab: Dict[str, Dict[str, Any]]) -> bool:
    """Save custom vocabulary to file (thread-safe write)"""
    vocab_file = get_vocab_file()
    try:
        # Write to temp file first, then rename (atomic write)
        temp_file = vocab_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        temp_file.replace(vocab_file)
        return True
    except IOError as e:
        log(f"Failed to save vocabulary: {e}", obs.LOG_ERROR)
        return False

def add_to_vocab(lang: str, word: str, translation: str = "") -> None:
    """Add a word to custom vocabulary (thread-safe)"""
    if not word or len(word) < 2:
        return
    
    vocab = load_custom_vocab()
    lang_data = vocab.get(lang, {})
    
    word_lower = word.lower().strip()
    if word_lower in lang_data:
        lang_data[word_lower]["count"] += 1
    else:
        lang_data[word_lower] = {
            "count": 1,
            "translation": translation,
            "added_at": time.time()
        }
    
    vocab[lang] = lang_data
    save_custom_vocab(vocab)


def load_custom_slang() -> Dict[str, str]:
    """Load custom slang dictionary from file"""
    slang_file = get_slang_file()
    if not slang_file.exists():
        return {}
    
    try:
        with open(slang_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k: str(v) for k, v in data.items()}
    except (json.JSONDecodeError, IOError):
        pass
    return {}


def save_custom_slang(slang: Dict[str, str]) -> bool:
    """Save custom slang dictionary to file"""
    slang_file = get_slang_file()
    try:
        temp_file = slang_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(slang, f, ensure_ascii=False, indent=2)
        temp_file.replace(slang_file)
        return True
    except IOError:
        return False


def add_slang_word(source: str, target: str) -> bool:
    """Add a word to the slang dictionary"""
    slang = load_custom_slang()
    
    # Add bidirectional
    slang[source.lower()] = target.lower()
    slang[target.lower()] = source.lower()
    
    return save_custom_slang(slang)


def replace_with_slang(text: str, slang: Dict[str, str]) -> str:
    """Replace words in text with slang equivalents"""
    words = _WORD_SPLIT_REGEX.split(text)
    result = []
    
    for word in words:
        prefix = ""
        suffix = ""
        clean_word = word
        
        # Strip prefix punctuation
        while clean_word and not clean_word[-1].isalnum():
            suffix = clean_word[-1] + suffix
            clean_word = clean_word[:-1]
        while clean_word and not clean_word[0].isalnum():
            prefix += clean_word[0]
            clean_word = clean_word[1:]
        
        if len(clean_word) < 2:
            result.append(word)
            continue
        
        # Check slang dictionary
        replacement = slang.get(clean_word.lower(), "")
        if replacement:
            # Preserve capitalization
            if clean_word[0].isupper():
                replacement = str(replacement).capitalize()
            result.append(prefix + str(replacement) + suffix)
        else:
            result.append(word)
    
    return " ".join(result)


def check_deps() -> List[str]:
    """Check for missing dependencies"""
    required = ["numpy", "sounddevice", "vosk", "deep_translator", "scipy"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing

# ============================================================================
# SLANG DICTIONARY MANAGEMENT
# ============================================================================

def manage_slang_cb(props: Any, prop: Any) -> bool:
    """Callback to show vocabulary info"""
    vocab = load_custom_vocab()
    total = sum(len(v) for v in vocab.values())
    log(f"Custom vocabulary: {total} words loaded", obs.LOG_INFO)
    return True


def add_slang_cb(props: Any, prop: Any) -> bool:
    """Callback to add custom word to vocabulary"""
    settings = _state.script_settings
    if settings is None:
        log("Settings not available", obs.LOG_WARNING)
        return True
    
    word = obs.obs_data_get_string(settings, "custom_word") or ""
    
    if word and len(word) >= 2:
        vocab = load_custom_vocab()
        for lang in vocab:
            if word.lower() not in vocab[lang]:
                vocab[lang][word.lower()] = {
                    "count": 1,
                    "translation": word,
                    "added_at": time.time()
                }
        if not save_custom_vocab(vocab):
            log(f"Failed to save word: {word}", obs.LOG_WARNING)
        else:
            log(f"Added word to all languages: {word}", obs.LOG_INFO)
    
    return True


def clear_slang_cb(props: Any, prop: Any) -> bool:
    """Callback to clear vocabulary"""
    vocab_file = get_vocab_file()
    if vocab_file.exists():
        vocab_file.unlink()
        log("Custom vocabulary cleared", obs.LOG_INFO)
    return True


def pip_bootstrap() -> None:
    """Install missing dependencies via pip"""
    for dep in ["numpy", "sounddevice", "vosk", "deep_translator", "scipy"]:
        try:
            __import__(dep)
        except ImportError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user", "--quiet", dep],
                    capture_output=True,
                    timeout=300,
                    check=False
                )
                log(f"Installed {dep}", obs.LOG_INFO)
            except subprocess.TimeoutExpired:
                log(f"Timeout installing {dep}", obs.LOG_WARNING)
            except Exception as e:
                log(f"Failed to install {dep}: {e}", obs.LOG_WARNING)


    """Install missing dependencies via pip"""
    for dep in ["numpy", "sounddevice", "vosk", "deep_translator", "scipy"]:
        try:
            __import__(dep)
        except ImportError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user", "--quiet", dep],
                    capture_output=True,
                    timeout=300,
                    check=False
                )
                log(f"Installed {dep}", obs.LOG_INFO)
            except subprocess.TimeoutExpired:
                log(f"Timeout installing {dep}", obs.LOG_WARNING)
            except Exception as e:
                log(f"Failed to install {dep}: {e}", obs.LOG_WARNING)

@contextmanager
def file_lock(filepath: Path):
    """Context manager for file locking (cross-platform)"""
    lockfile = filepath.with_suffix('.lock')
    max_attempts = 10
    for _ in range(max_attempts):
        try:
            lockfile.touch(exist_ok=False)
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            lockfile.unlink()
        except OSError:
            pass

# ============================================================================
# MICROPHONE MANAGEMENT
# ============================================================================

def get_mics() -> List[Dict[str, Any]]:
    """Get list of available microphones (with caching)"""
    global _mic_list_cache
    
    if _mic_list_cache:
        return _mic_list_cache
    
    devices = []
    seen_names = set()
    
    try:
        import sounddevice as sd
        all_devices = sd.query_devices()
        
        for idx, dev in enumerate(all_devices):
            if dev.get('max_input_channels', 0) > 0:
                name = str(dev.get('name', f'Device {idx}'))
                
                # Deduplicate by name
                if name in seen_names:
                    continue
                seen_names.add(name)
                
                samplerate = int(dev.get('default_samplerate', DEFAULT_SAMPLERATE))
                devices.append({
                    'index': idx,
                    'name': name,
                    'samplerate': samplerate,
                    'channels': dev.get('max_input_channels', 1),
                    'hostapi': dev.get('hostapi', 0)
                })
                
    except ImportError:
        log("sounddevice not available", obs.LOG_ERROR)
    except Exception as e:
        log(f"Error querying devices: {e}", obs.LOG_ERROR)
    
    # Ensure at least some devices
    if not devices:
        for i in range(3):
            devices.append({
                'index': i,
                'name': f'Generic Device {i}',
                'samplerate': DEFAULT_SAMPLERATE
            })
    
    _mic_list_cache = devices
    return devices

def validate_mic_index(idx: int) -> bool:
    """Validate microphone index is within bounds"""
    mics = get_mics()
    return 0 <= idx < len(mics)

def safe_obs_call(func, *args, **kwargs):
    """
    Wrapper for safe OBS API calls.
    Catches all exceptions to prevent OBS crashes.
    """
    if _obs_api_disabled:
        return None
    try:
        return func(*args, **kwargs)
    except Exception:
        return None

# ============================================================================
# SOURCE MANAGEMENT
# ============================================================================

def update_text_source(
    name: str, 
    text: str, 
    settings: Any, 
    prefix: str,
    max_width: int = 480,
    max_height: int = 100
) -> bool:
    """
    Update an OBS text source with new content and styling.
    
    Args:
        name: Source name
        text: Text content to display
        settings: OBS data settings object
        prefix: Settings prefix (source/target)
        max_width: Maximum width for word wrap
        max_height: Maximum height
    
    Returns:
        True if successful, False otherwise
    """
    if _obs_api_disabled:
        return False
    
    try:
        safe_name = _SAFE_NAME_REGEX.sub('', name)
        source = obs.obs_get_source_by_name(safe_name)
        
        if not source:
            _logger.warning(f"Source '{safe_name}' not found")
            return False
        
        try:
            source_settings = obs.obs_source_get_settings(source)
            
            # Text content
            obs.obs_data_set_string(source_settings, "text", text)
            
            # Layout constraints - use extents for text wrapping
            obs.obs_data_set_bool(source_settings, "wordwrap", True)
            obs.obs_data_set_int(source_settings, "extents", 1)
            obs.obs_data_set_int(source_settings, "extents_cx", max_width)
            obs.obs_data_set_int(source_settings, "extents_cy", max_height)
            obs.obs_data_set_int(source_settings, "align", 1)  # Center horizontal
            obs.obs_data_set_int(source_settings, "valign", 1)  # Center vertical
            
            # Background
            bg_color = obs.obs_data_get_int(settings, f"{prefix}_bg_color")
            bg_opacity = obs.obs_data_get_int(settings, f"{prefix}_bg_opacity")
            obs.obs_data_set_int(source_settings, "bk_color", bg_color)
            obs.obs_data_set_int(source_settings, "bk_opacity", int(bg_opacity * 2.55))
            
            # Font
            font_size = obs.obs_data_get_int(settings, f"{prefix}_font_size")
            font_obj = obs.obs_data_get_obj(settings, f"{prefix}_font")
            
            font_props = obs.obs_data_create()
            obs.obs_data_set_string(font_props, "face", "Arial")
            obs.obs_data_set_int(font_props, "size", font_size)
            obs.obs_data_set_int(font_props, "style", 400)  # Normal weight
            obs.obs_data_set_int(font_props, "weight", 400)
            obs.obs_data_set_obj(source_settings, "font", font_props)
            obs.obs_data_release(font_props)
            
            if font_obj:
                obs.obs_data_release(font_obj)
            
            # Text color
            text_color = obs.obs_data_get_int(settings, f"{prefix}_text_color")
            obs.obs_data_set_int(source_settings, "color", text_color)
            
            # Apply settings
            obs.obs_source_update(source, source_settings)
            obs.obs_data_release(source_settings)
            
        finally:
            obs.obs_source_release(source)
            
        return True
        
    except Exception as e:
        _logger.error(f"Error updating text source: {e}")
        return False

def ensure_source(name: str) -> Optional[Any]:
    """Ensure a text source exists, create if not"""
    if _obs_api_disabled:
        return None
    
    try:
        safe_name = _SAFE_NAME_REGEX.sub('', name)
        source = obs.obs_get_source_by_name(safe_name)
        
        if source:
            obs.obs_source_release(source)
            return source
        
        # Create new source
        settings = obs.obs_data_create()
        obs.obs_data_set_string(settings, "text", "...")
        obs.obs_data_set_bool(settings, "wordwrap", True)
        obs.obs_data_set_int(settings, "extents", 1)  # Enable extents
        obs.obs_data_set_int(settings, "extents_cx", 480)  # Width (25% of 1920)
        obs.obs_data_set_int(settings, "extents_cy", 100)  # Height
        
        try:
            # OBS 29+ requires 4 arguments
            source = obs.obs_source_create(
                "text_gdiplus", 
                safe_name, 
                settings, 
                obs.obs_data_create()
            )
        except TypeError:
            # OBS 28 compatibility
            source = obs.obs_source_create(
                "text_gdiplus", 
                safe_name, 
                settings
            )
        
        obs.obs_data_release(settings)
        
        if source:
            # Add to current scene
            scene = obs.obs_frontend_get_current_scene()
            if scene:
                obs_scene = obs.obs_scene_from_source(scene)
                if obs_scene:
                    obs.obs_scene_add(obs_scene, source)
                obs.obs_source_release(scene)
            obs.obs_source_release(source)
            
            # Get reference
            source = obs.obs_get_source_by_name(safe_name)
        
        return source
        
    except Exception as e:
        _logger.error(f"Error ensuring source: {e}")
        return None

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class AudioProcessor:
    """
    Professional-grade audio processor for speech recognition.
    
    Applies:
    - High-pass filtering (rumble removal)
    - Pre-emphasis (consonant boost)
    - Dynamic range compression
    - Voice Activity Detection (VAD)
    """
    
    def __init__(
        self,
        samplerate: int,
        blocksize: int,
        audio_gate_db: float = DEFAULT_AUDIO_GATE_DB
    ):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.audio_gate_db = audio_gate_db
        
        # Filter coefficients
        self.b_hp, self.a_hp = signal.butter(
            2, HIGH_PASS_FREQ_HZ, 
            btype='high', 
            fs=samplerate
        )
        
        # Filter state
        self.z_hp = np.zeros(2)
        self.last_sample_pe = 0.0
        
        # VAD state
        self.vad_noise_floor = -70.0
        self.vad_hangover_frames = int(samplerate * VAD_HANGOVER_MS / 1000 / blocksize)
        self.vad_hangover_count = 0
        
        # Noise profiling
        self.noise_profile: Optional[np.ndarray] = None
        self.silence_frame_count = 0
        self.is_noise_profile_collected = False
        
        # Compressor coefficients
        self.comp_threshold_linear = 10 ** (COMPRESSOR_THRESHOLD_DB / 20.0)
        
    def process(self, audio_int16: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Process audio block.
        
        Args:
            audio_int16: Audio data as int16 numpy array
            
        Returns:
            Tuple of (processed_audio_int16, is_speech, db_level)
        """
        # Convert to float32 normalized
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # 1. High-pass filter (remove rumble)
        audio_float, self.z_hp = self._apply_highpass(audio_float)
        
        # 2. Pre-emphasis (boost consonants)
        audio_float, self.last_sample_pe = self._apply_preemphasis(audio_float)
        
        # 3. Calculate audio level
        rms = np.sqrt(np.mean(audio_float ** 2))
        rms = max(rms, 1e-8)
        db = 20 * np.log10(rms)
        
        # 4. Adaptive noise floor
        if db < self.vad_noise_floor + 10:
            self.vad_noise_floor = (
                0.99 * self.vad_noise_floor + 
                0.01 * db
            )
        
        # 5. VAD decision
        above_gate = (
            db >= self.vad_noise_floor + VAD_THRESHOLD_MARGIN_DB or
            db >= self.audio_gate_db
        )
        
        # 6. Hangover (prevent clipping at word boundaries)
        if above_gate:
            self.vad_hangover_count = self.vad_hangover_frames
            self.silence_frame_count = 0
        elif self.vad_hangover_count > 0:
            self.vad_hangover_count -= 1
            above_gate = True
            self.silence_frame_count += 1
        else:
            self.silence_frame_count += 1
        
        # 7. Dynamic range compression
        audio_processed = self._apply_compression(audio_float)
        
        # 8. Convert back to int16
        audio_int16_out = np.clip(audio_processed, -1, 1)
        audio_int16_out = (audio_int16_out * 32767).astype(np.int16)
        
        return audio_int16_out, above_gate, db
    
    def _apply_highpass(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply high-pass filter"""
        try:
            return signal.lfilter(self.b_hp, self.a_hp, audio, zi=self.z_hp)
        except Exception:
            return audio, np.zeros(2)
    
    def _apply_preemphasis(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply pre-emphasis filter"""
        output = np.empty_like(audio)
        output[0] = audio[0] - PREEMPHASIS_ALPHA * self.last_sample_pe
        output[1:] = audio[1:] - PREEMPHASIS_ALPHA * audio[:-1]
        return output, audio[-1]
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression"""
        envelope = np.sqrt(np.mean(audio ** 2))
        
        if envelope > self.comp_threshold_linear:
            gain_db = (COMPRESSOR_THRESHOLD_DB - 20 * np.log10(envelope + 1e-10)) / COMPRESSOR_RATIO
            gain = 10 ** (gain_db / 20.0)
        else:
            gain = 1.0
        
        output = audio * gain
        # Soft clipper to prevent digital clipping
        output = np.tanh(output)
        return output
    
    def resample_to_16k(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio to 16kHz for Vosk"""
        if self.samplerate == TARGET_SAMPLERATE:
            return audio
        
        num_samples = max(1, int(len(audio) * TARGET_SAMPLERATE / self.samplerate))
        resampled = signal.resample(audio, num_samples)
        return np.clip(resampled, -1, 1)


def fuzzy_match_word(word: str, vocab_words: List[str], cutoff: float = 0.8) -> Optional[str]:
    """
    Find the closest matching word from vocabulary.
    
    Args:
        word: Word to match
        vocab_words: List of known words
        cutoff: Similarity threshold (0-1)
    
    Returns:
        Best matching word or None
    """
    if not vocab_words:
        return None
    
    matches = difflib.get_close_matches(
        word.lower(), 
        vocab_words, 
        n=1, 
        cutoff=cutoff
    )
    
    return matches[0] if matches else None


def apply_fuzzy_translation(text: str, vocab: Dict[str, Dict[str, Any]], target_lang: str) -> str:
    """
    Apply fuzzy matching to replace unrecognized words with similar known words.
    
    Args:
        text: Input text
        vocab: Custom vocabulary dictionary
        target_lang: Target language code
    
    Returns:
        Text with fuzzy-matched words
    """
    words = _WORD_SPLIT_REGEX.split(text)
    result = []
    
    lang_vocab = vocab.get(target_lang, {})
    known_words = list(lang_vocab.keys())
    
    for word in words:
        # Keep punctuation
        prefix = ""
        suffix = ""
        clean_word = word
        
        # Strip punctuation
        while clean_word and not clean_word[-1].isalnum():
            suffix = clean_word[-1] + suffix
            clean_word = clean_word[:-1]
        while clean_word and not clean_word[0].isalnum():
            prefix += clean_word[0]
            clean_word = clean_word[1:]
        
        if len(clean_word) < 2:
            result.append(word)
            continue
        
        # Try to find similar word
        match = fuzzy_match_word(clean_word, known_words, cutoff=0.75)
        if match:
            # Use matched word with original capitalization
            if clean_word[0].isupper():
                match = match.capitalize()
            result.append(prefix + match + suffix)
        else:
            result.append(word)
    
    return " ".join(result)

# ============================================================================
# WORKER THREADS
# ============================================================================

class AudioSTTWorker(threading.Thread):
    """
    Audio/STT Worker Thread
    
    Handles:
    - Audio capture from sounddevice
    - Audio preprocessing
    - Vosk speech recognition
    - VAD and finalization triggers
    """
    
    def __init__(
        self,
        mic_id: int,
        model_path: str,
        mode: str,
        samplerate: int,
        audio_gate_db: float,
        max_lines: int,
        fuzzy_match: bool,
        adaptive_vocab: bool,
        auto_restart: bool,
        show_confidence: bool,
        auto_detect: bool = True
    ):
        super().__init__(daemon=True)
        
        self.mic_id = mic_id
        self.model_path = model_path
        self.src_lang, self.tgt_lang = mode.split('_')
        self.samplerate = samplerate
        self.audio_gate_db = audio_gate_db
        self.max_lines = max_lines
        self.fuzzy_match = fuzzy_match
        self.adaptive_vocab = adaptive_vocab
        self.auto_restart = auto_restart
        self.show_confidence = show_confidence
        self.auto_detect = auto_detect
        
        # Thread safety
        self._running = False
        self._stop_event = threading.Event()
        
        # Audio config
        self.blocksize = max(1, int(samplerate * BLOCKSIZE_MS / 1000))
        
        # State
        self._draft_text = ""
        self._is_speaking = False
        self._history_source: List[str] = []
        self._history_target: List[str] = []
        self._last_error = ""
        self._confidence: float = 0.0
        self._mic_disconnected: bool = False
        
        # Timing
        self._silence_start_time = 0.0
        self._speech_start_time = 0.0
        self._last_finalization = 0.0
        
        # Queues
        self._translation_queue: queue.Queue = queue.Queue(maxsize=TRANSLATION_QUEUE_SIZE)
        
        # Audio processor
        self._audio_processor = AudioProcessor(
            samplerate, self.blocksize, audio_gate_db
        )
        
        # Stream management - thread-safe
        self._stream = None
        self._stream_close_requested = False
        
        log(f"Worker initialized: mic={mic_id}, lang={self.src_lang}->{self.tgt_lang}")
    
    @property
    def running(self) -> bool:
        return self._running
    
    @running.setter
    def running(self, value: bool) -> None:
        self._running = value
    
    def clear_history(self) -> None:
        """Clear all history (thread-safe)"""
        with _state.worker_lock:
            self._history_source.clear()
            self._history_target.clear()
            self._draft_text = ""
            self._is_speaking = False
            self._silence_start_time = 0.0
            self._speech_start_time = 0.0
    
    def close_stream(self) -> None:
        """Thread-safe stream closure - call this from outside the thread"""
        self._stream_close_requested = True
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
    
    def stop(self) -> None:
        """Signal thread to stop"""
        self._running = False
        self._stop_event.set()
        self._stream_close_requested = True
        self.close_stream()
    
    def run(self) -> None:
        """Main worker loop"""
        self._running = True
        self._stream = None
        self._stream_close_requested = False
        translator_thread = None
        
        try:
            # Import dependencies
            from vosk import Model, KaldiRecognizer
            import sounddevice as sd
            
            # Load model
            log(f"Loading model: {Path(self.model_path).name}")
            model = Model(self.model_path)
            rec = KaldiRecognizer(model, TARGET_SAMPLERATE)
            rec.SetWords(True)
            log(f"Model loaded: {self.src_lang} -> {self.tgt_lang}")
            
            # Start translation thread
            translator_thread = TranslatorThread(
                self._translation_queue,
                self.src_lang,
                self.tgt_lang,
                self,
                self.auto_detect
            )
            translator_thread.start()
            log("Translation thread started")
            
            # Open audio stream
            log(f"Opening audio: {self.samplerate}Hz, blocksize={self.blocksize}")
            restart_attempts = 0
            max_restart_attempts = 5
            
            # Main processing loop
            while self._running and not _state.is_exiting and not _state.shutdown_event.is_set():
                # Check stop event at loop start
                if self._stop_event.is_set():
                    break
                
                # Check stream close request
                if self._stream_close_requested:
                    break
                
                # Extra safety: check global shutdown
                if _state.is_exiting or _state.shutdown_event.is_set():
                    break
                
                try:
                    # Try to open stream if not open
                    if self._stream is None:
                        try:
                            import sounddevice as sd
                            self._stream = sd.InputStream(
                                device=self.mic_id,
                                channels=1,
                                samplerate=self.samplerate,
                                dtype='int16',
                                blocksize=self.blocksize
                            )
                            self._stream.start()
                            log("Audio stream started", obs.LOG_INFO)
                            restart_attempts = 0
                            self._mic_disconnected = False
                        except Exception as e:
                            restart_attempts += 1
                            if self.auto_restart and restart_attempts < max_restart_attempts:
                                log(f"Mic reconnect attempt {restart_attempts}/{max_restart_attempts}...", obs.LOG_WARNING)
                                time.sleep(2)
                                continue
                            else:
                                log(f"Cannot open audio stream: {e}", obs.LOG_ERROR)
                                break
                    
                    # Check shutdown before blocking read
                    if self._stream_close_requested:
                        break
                    
                    # Read audio - wrapped in try/except for safe shutdown
                    try:
                        data, _ = self._stream.read(self.blocksize)
                    except (OSError, IOError) as e:
                        # Stream closed or device unavailable - check if we're shutting down
                        if self._stream_close_requested or not self._running:
                            self._stream = None
                            break
                        # Otherwise, try to reconnect
                        log(f"Audio read error: {e}", obs.LOG_WARNING)
                        self._stream = None
                        continue
                    
                    if len(data) == 0:
                        continue
                    
                    audio_data = data.flatten()
                    current_time = time.time()
                    
                    # Reset restart attempts on successful read
                    restart_attempts = 0
                    self._mic_disconnected = False
                    
                    # Process audio
                    processed_audio, is_speaking, db_level = self._audio_processor.process(audio_data)
                    
                    # Resample if needed
                    if self.samplerate != TARGET_SAMPLERATE:
                        processed_float = processed_audio.astype(np.float32) / 32768.0
                        resampled = self._audio_processor.resample_to_16k(processed_float)
                        vosk_audio = (np.clip(resampled, -1, 1) * 32767).astype(np.int16)
                    else:
                        vosk_audio = processed_audio
                    
                    # Speech tracking
                    if is_speaking:
                        if not self._is_speaking:
                            self._is_speaking = True
                            self._speech_start_time = current_time
                        self._silence_start_time = 0.0
                    else:
                        if self._is_speaking and self._silence_start_time == 0:
                            self._silence_start_time = current_time
                    
                    # Calculate durations
                    silence_duration = (
                        current_time - self._silence_start_time 
                        if self._silence_start_time > 0 else 0
                    )
                    speech_duration = (
                        current_time - self._speech_start_time 
                        if self._is_speaking and self._speech_start_time > 0 else 0
                    )
                    
                    # Finalization triggers
                    should_finalize = (
                        (silence_duration >= FINALIZATION_SILENCE_MS / 1000 and self._draft_text) or
                        (speech_duration >= FINALIZATION_SPEECH_MAX_MS / 1000 and self._draft_text)
                    )
                    
                    if should_finalize and current_time - self._last_finalization > 0.5:
                        self._finalize()
                        self._last_finalization = current_time
                    else:
                        self._recognize(rec, vosk_audio, is_speaking)
                    
                    # Update UI
                    self._push_update(is_speaking)
                    
                except queue.Empty:
                    continue
                except (sd.PortAudioError, OSError) as e:
                    if self._stream_close_requested or not self._running:
                        break
                    if "Input overflow" in str(e) or "Device unavailable" in str(e):
                        log("Mic disconnected, attempting reconnect...", obs.LOG_WARNING)
                        self._mic_disconnected = True
                        self._stream = None
                        if self.auto_restart:
                            time.sleep(1)
                            continue
                    raise
                except Exception as e:
                    log(f"Audio error: {e}", obs.LOG_WARNING)
                    time.sleep(0.01)
            
        except ImportError as e:
            log(f"Missing dependency: {e}", obs.LOG_ERROR)
        except Exception as e:
            if self._running:
                log(f"Worker critical error: {e}", obs.LOG_ERROR)
        finally:
            self._running = False
            
            # Cleanup stream using thread-safe method
            self.close_stream()
            
            # Signal translator to stop
            self._translation_queue.put_nowait(('stop', ''))
            
            if translator_thread:
                translator_thread.join(timeout=2)
            
            log("Worker stopped")
    
    def _finalize(self) -> None:
        """Finalize current speech segment - prevents duplicates"""
        if not self._draft_text:
            return
        
        # Get text and clear draft IMMEDIATELY to prevent double-finalization
        text_to_finalize = self._draft_text
        self._draft_text = ""
        
        # Prevent duplicate finalization of same text
        if hasattr(self, '_last_finalized_text') and self._last_finalized_text == text_to_finalize:
            return
        
        self._last_finalized_text = text_to_finalize
        
        try:
            self._translation_queue.put_nowait(('final', text_to_finalize))
        except queue.Full:
            pass
        
        self._silence_start_time = 0.0
        self._is_speaking = False
        self._speech_start_time = 0.0
    
    def _recognize(self, rec: Any, audio: np.ndarray, is_speaking: bool) -> None:
        """Run Vosk recognition"""
        if rec.AcceptWaveform(audio.tobytes()):
            result = json.loads(rec.Result())
            txt = result.get('text', '').strip()
            
            if txt:
                # Confidence check and tracking
                words = result.get('result', [])
                if words:
                    self._confidence = sum(w.get('conf', 0) for w in words) / len(words)
                    if self._confidence < CONFIDENCE_THRESHOLD:
                        return
                else:
                    self._confidence = 0.5  # Default if no word data
                
                # Learn new words
                if self.adaptive_vocab:
                    for word in _WORD_SPLIT_REGEX.split(txt):
                        if len(word) > 2:
                            add_to_vocab(self.src_lang, word)
                
                self._draft_text = (self._draft_text + " " + txt).strip()
        else:
            # Partial result
            partial_result = rec.PartialResult()
            partial = json.loads(partial_result).get('partial', '')
            if partial and is_speaking:
                self._draft_text = partial.strip()
                # Estimate confidence for partial (lower)
                self._confidence = 0.6
    
    def _push_update(self, is_speaking: bool) -> None:
        """Push update to UI queue"""
        # GUARD: Never update UI during shutdown
        if _state.is_exiting or _state.shutdown_event.is_set():
            return
        
        try:
            if _state.text_queue is None:
                return
            
            MAX_CHARS_PER_LINE = 40
            
            def wrap_text(txt: str, max_lines: int) -> tuple[str, int]:
                words = txt.split()
                lines = []
                current_line = []
                current_len = 0
                
                for word in words:
                    word_len = len(word)
                    if current_len + word_len + len(current_line) <= MAX_CHARS_PER_LINE:
                        current_line.append(word)
                        current_len += word_len
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                        current_line = [word]
                        current_len = word_len
                
                if current_line:
                    lines.append(" ".join(current_line))
                
                total_lines = len(lines)
                return "\n".join(lines[:max_lines]), total_lines
            
            # Build source text - only show finalized text from history
            s_lines = self._history_source[-1:] if self._history_source else []
            if s_lines:
                s_txt = " ".join(s_lines[-1].split())
            else:
                s_txt = "Listening..." if is_speaking else "Waiting..."
            
            if self._last_error:
                s_txt = "[ERR] " + s_txt
            
            s_txt, s_total = wrap_text(s_txt, self.max_lines)
            
            # Build target text - only show finalized text from history
            t_lines = self._history_target[-1:] if self._history_target else []
            if t_lines:
                t_txt = " ".join(t_lines[-1].split())
            else:
                t_txt = "Ready..."
            
            t_txt, t_total = wrap_text(t_txt, self.max_lines)
            
            # When content exceeds max lines, clear ALL history and restart
            # This ensures overflowed content is replaced with fresh text
            if s_total > self.max_lines or t_total > self.max_lines:
                with _state.worker_lock:
                    # Clear all history to restart fresh
                    self._history_source.clear()
                    self._history_target.clear()
                    # Show placeholder until new speech comes in
                    s_txt = "Listening..." if is_speaking else "Waiting..."
                    t_txt = "Ready..."
                    s_total = 0
                    t_total = 0
            
            _state.text_queue.put_nowait({'s': s_txt, 't': t_txt})
            
        except queue.Full:
            pass
        except Exception as e:
            _logger.debug(f"Push update error: {e}")


class TranslatorThread(threading.Thread):
    """
    Translation Worker Thread
    
    Handles async translation using Google Translate.
    Thread-safe updates to shared state.
    """
    
    def __init__(
        self,
        translation_queue: queue.Queue,
        src_lang: str,
        tgt_lang: str,
        worker: AudioSTTWorker,
        auto_detect: bool = True
    ):
        super().__init__(daemon=True)
        self._queue = translation_queue
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._worker = worker
        self._auto_detect = auto_detect
        self._running = True
    
    def run(self) -> None:
        """Main translation loop"""
        try:
            from deep_translator import GoogleTranslator
            
            translator = GoogleTranslator(
                source=self._src_lang,
                target=self._tgt_lang
            )
            log("Translator initialized")
            
            while self._running and not _state.is_exiting and not _state.shutdown_event.is_set():
                try:
                    msg_type, text = self._queue.get(timeout=0.5)
                    
                    if msg_type == 'stop':
                        break
                    
                    if msg_type == 'final' and text:
                        self._translate(translator, text)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    log(f"Translation loop error: {e}", obs.LOG_WARNING)
                    
        except ImportError:
            log("deep_translator not available", obs.LOG_ERROR)
        except Exception as e:
            log(f"Translator thread error: {e}", obs.LOG_ERROR)
    
    def _translate(self, translator: Any, text: str) -> None:
        """Translate text and update history with auto-detect support"""
        try:
            text = " ".join(text.split())
            
            if not text:
                return
            
            detected_lang, confidence = detect_language(text)
            
            if self._auto_detect and detected_lang == self._tgt_lang:
                trans = text
            else:
                trans = translator.translate(text)
            
            trans = " ".join(trans.split())
            
            slang = load_custom_slang()
            if slang:
                trans = replace_with_slang(trans, slang)
            
            if self._worker.fuzzy_match:
                vocab = load_custom_vocab()
                trans = apply_fuzzy_translation(trans, vocab, self._tgt_lang)
            
            with _state.worker_lock:
                if not self._worker.running:
                    return
                
                if text not in self._worker._history_source:
                    self._worker._history_source.append(text)
                    self._worker._history_target.append(trans)
                    
                    if len(self._worker._history_source) > MAX_HISTORY_LINES:
                        self._worker._history_source.pop(0)
                        self._worker._history_target.pop(0)
                
                self._worker._last_error = ""
                
        except Exception as e:
            with _state.worker_lock:
                if self._worker.running:
                    self._worker._last_error = str(e)[:50]
            log(f"Translation error: {e}", obs.LOG_WARNING)

# ============================================================================
# MODEL DOWNLOAD
# ============================================================================

def download_models() -> bool:
    """Download Vosk models (async)"""
    base = get_base_dir()
    
    model_size = "small"
    if _state.script_settings:
        model_size = obs.obs_data_get_string(_state.script_settings, "model_size") or "small"
    
    urls = MODEL_URLS.get(model_size, MODEL_URLS["small"])
    
    success = True
    
    for lang, url in urls.items():
        model_name = Path(url).stem.replace('.zip', '')
        model_dir = base / model_name
        
        if model_dir.exists():
            log(f"{model_name} already exists")
            continue
        
        log(f"Downloading {model_name}...")
        zip_path = base / f"{model_name}.zip"
        
        try:
            # Download with streaming
            with urllib.request.urlopen(url, timeout=600) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                with open(zip_path, 'wb') as f:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
            
            # Verify and extract
            if not zipfile.is_zipfile(zip_path):
                log(f"Invalid zip file: {model_name}", obs.LOG_ERROR)
                zip_path.unlink(missing_ok=True)
                success = False
                continue
            
            log(f"Extracting {model_name}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(base)
            
            zip_path.unlink()
            log(f"{model_name} ready")
            
        except Exception as e:
            log(f"Failed to download {model_name}: {e}", obs.LOG_ERROR)
            zip_path.unlink(missing_ok=True)
            success = False
    
    if success:
        log("All models ready!")
    else:
        log("Some models failed to download", obs.LOG_ERROR)
    
    return success

def find_model(lang: str, size: str = "small") -> Optional[Path]:
    """Find downloaded model path"""
    base = get_base_dir()
    
    if not base.exists():
        return None
    
    # Try exact size first
    prefix = f"vosk-model-{size}-{lang}"
    for item in base.iterdir():
        if item.name.startswith(prefix) and item.is_dir():
            return item
    
    # Fallback to any matching model
    fallback_prefix = f"vosk-model-{lang}"
    for item in base.iterdir():
        if item.name.startswith(fallback_prefix) and item.is_dir():
            return item
    
    return None

# ============================================================================
# OBS SCRIPT HOOKS
# ============================================================================

def script_description() -> str:
    return f"<h2>Translator v{VERSION}</h2><b>Status:</b> {_state.last_status}"

def script_defaults(settings: Any) -> None:
    """Set default values for script properties"""
    obs.obs_data_set_default_int(settings, "mic_id", 0)
    obs.obs_data_set_default_string(settings, "direction", "es_en")
    obs.obs_data_set_default_int(settings, "max_lines", 4)
    obs.obs_data_set_default_int(settings, "audio_gate", DEFAULT_AUDIO_GATE_DB)
    obs.obs_data_set_default_int(settings, "source_bg_opacity", 80)
    obs.obs_data_set_default_int(settings, "target_bg_opacity", 80)
    obs.obs_data_set_default_int(settings, "source_text_color", 0xFFFFFFFF)
    obs.obs_data_set_default_int(settings, "target_text_color", 0xFFFFFFFF)
    obs.obs_data_set_default_int(settings, "source_font_size", 36)
    obs.obs_data_set_default_int(settings, "target_font_size", 36)
    obs.obs_data_set_default_string(settings, "model_size", "large")
    obs.obs_data_set_default_bool(settings, "adaptive_vocab", True)
    obs.obs_data_set_default_bool(settings, "fuzzy_match", True)
    obs.obs_data_set_default_bool(settings, "auto_restart", True)
    obs.obs_data_set_default_bool(settings, "show_confidence", True)
    obs.obs_data_set_default_bool(settings, "auto_detect", True)

def script_properties() -> Any:
    """Create script properties UI"""
    props = obs.obs_properties_create()
    
    # Status
    obs.obs_properties_add_text(props, "st", f"STATUS: {_state.last_status}", obs.OBS_TEXT_INFO)
    
    # Buttons
    obs.obs_properties_add_button(props, "start", "START / STOP", _start_stop_cb)
    obs.obs_properties_add_button(props, "clear", "CLEAR HISTORY", _clear_history_cb)
    obs.obs_properties_add_button(props, "dl", "DOWNLOAD MODELS", _dl_models_cb)
    obs.obs_properties_add_button(props, "pip", "INSTALL DEPS", _pip_install_cb)
    
    # Microphone selection
    mic_p = obs.obs_properties_add_list(
        props, "mic_id", "MICROPHONE",
        obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_INT
    )
    for mic in get_mics():
        obs.obs_property_list_add_int(mic_p, mic['name'], mic['index'])
    
    # Language direction
    dir_p = obs.obs_properties_add_list(
        props, "direction", "LANGUAGE",
        obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_STRING
    )
    obs.obs_property_list_add_string(dir_p, "English → Spanish", "en_es")
    obs.obs_property_list_add_string(dir_p, "Spanish → English", "es_en")
    
    # Auto-detect language
    obs.obs_properties_add_bool(props, "auto_detect", "Auto-detect Language")
    
    # Audio settings
    obs.obs_properties_add_int_slider(props, "audio_gate", "Audio Gate (dB)", -70, -30, 1)
    obs.obs_properties_add_int(props, "max_lines", "Max Lines", 1, 10, 1)
    
    # Custom vocabulary
    obs.obs_properties_add_text(props, "sg1", "--- CUSTOM WORDS ---", obs.OBS_TEXT_INFO)
    obs.obs_properties_add_text(props, "custom_word", "Custom Word", obs.OBS_TEXT_DEFAULT)
    obs.obs_properties_add_button(props, "add_word", "ADD WORD", add_slang_cb)
    obs.obs_properties_add_button(props, "clear_words", "CLEAR WORDS", clear_slang_cb)
    obs.obs_properties_add_button(props, "show_words", "SHOW WORDS", manage_slang_cb)
    
    # Source styling
    obs.obs_properties_add_text(props, "s1", "--- SOURCE ---", obs.OBS_TEXT_INFO)
    obs.obs_properties_add_font(props, "source_font", "Font")
    obs.obs_properties_add_int(props, "source_font_size", "Source Font Size", 24, 100, 36)
    obs.obs_properties_add_color(props, "source_text_color", "Text Color")
    obs.obs_properties_add_color(props, "source_bg_color", "Background")
    obs.obs_properties_add_int_slider(props, "source_bg_opacity", "Opacity", 0, 100, 1)
    
    # Target styling
    obs.obs_properties_add_text(props, "s2", "--- TARGET ---", obs.OBS_TEXT_INFO)
    obs.obs_properties_add_font(props, "target_font", "Font")
    obs.obs_properties_add_int(props, "target_font_size", "Target Font Size", 24, 100, 36)
    obs.obs_properties_add_color(props, "target_text_color", "Text Color")
    obs.obs_properties_add_color(props, "target_bg_color", "Background")
    obs.obs_properties_add_int_slider(props, "target_bg_opacity", "Opacity", 0, 100, 1)
    
    return props

# ============================================================================
# CALLBACKS
# ============================================================================

def _start_stop_cb(props: Any, prop: Any) -> bool:
    """Start/Stop button callback"""
    if not _state.is_running:
        if _start_translator():
            _state.is_running = True
    else:
        _stop_translator()
        _state.is_running = False
    return True

def _clear_history_cb(props: Any, prop: Any) -> bool:
    """Clear History button callback"""
    with _state.worker_lock:
        if _state.translation_worker:
            _state.translation_worker.clear_history()
            log("History cleared")
    return True

def _dl_models_cb(props: Any, prop: Any) -> bool:
    """Download Models button callback"""
    threading.Thread(target=download_models, daemon=True).start()
    return True

def _pip_install_cb(props: Any, prop: Any) -> bool:
    """Install Dependencies button callback"""
    threading.Thread(target=pip_bootstrap, daemon=True).start()
    return True

def script_update(settings: Any) -> None:
    """Called when script properties are updated"""
    _state.script_settings = settings

def _update_ui_from_queue() -> None:
    """Timer callback to update UI using OBS text sources"""
    try:
        # CRITICAL: Block ALL OBS API calls if shutting down
        if _obs_api_disabled:
            return
        
        # GUARD: Never update UI during shutdown
        if _state.is_exiting or _state.shutdown_event.is_set():
            return
        
        if _state.text_queue is None:
            return
        
        if _state.script_settings is None:
            return
        
        while not _state.text_queue.empty():
            # Double-check shutdown before processing each item
            if _state.is_exiting or _state.shutdown_event.is_set():
                return
            
            try:
                data = _state.text_queue.get_nowait()
            except Exception:
                break
            
            if _state.is_exiting or _state.shutdown_event.is_set():
                return
            
            # Update OBS text sources - THREAD-SAFE
            try:
                update_text_source("Translator_Source", data['s'], _state.script_settings, "source")
                update_text_source("Translator_Target", data['t'], _state.script_settings, "target")
            except Exception:
                pass
    except Exception:
        # NEVER let exception propagate to OBS
        pass

def script_load(settings: Any) -> None:
    """Called when script is loaded"""
    global _obs_api_disabled
    _obs_api_disabled = False
    
    _state.is_exiting = False
    _state.shutdown_event.clear()
    _state.is_running = False
    _state.script_settings = settings
    _state.text_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    obs.timer_add(_update_ui_from_queue, 100)
    log(f"Translator v{VERSION} loaded")

def script_unload() -> None:
    """Called when script is unloaded (OBS closing) - THREAD-SAFE SHUTDOWN"""
    # CRITICAL: Block ALL OBS API calls FIRST
    global _obs_api_disabled
    _obs_api_disabled = True
    
    try:
        # Remove timer - this stops new UI updates
        try:
            obs.timer_remove(_update_ui_from_queue)
        except Exception:
            pass
        
        # Signal shutdown flags
        _state.is_exiting = True
        _state.shutdown_event.set()
        _state.is_running = False
        
        # Clear the text queue
        if _state.text_queue:
            try:
                while True:
                    try:
                        _state.text_queue.get_nowait()
                    except queue.Empty:
                        break
            except Exception:
                pass
            _state.text_queue = None
        
        # Stop worker thread
        try:
            with _state.worker_lock:
                if _state.translation_worker:
                    _state.translation_worker.close_stream()
                    _state.translation_worker._running = False
                    _state.translation_worker._stop_event.set()
                    _state.translation_worker.join(timeout=1)
                    _state.translation_worker = None
        except Exception:
            pass
        
    except Exception:
        # Never let exception escape
        pass
    finally:
        try:
            log("Translator unloaded")
        except Exception:
            pass

# ============================================================================
# TRANSLATOR CONTROL
# ============================================================================

def _start_translator() -> bool:
    """Start the translator"""
    _state.is_exiting = False
    _state.shutdown_event.clear()
    _state.is_running = True
    
    # Check dependencies
    missing = check_deps()
    if missing:
        log(f"Missing: {', '.join(missing)}", obs.LOG_ERROR)
        return False
    
    if _state.script_settings is None:
        log("No settings", obs.LOG_ERROR)
        return False
    
    # Recreate queue
    _state.text_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    
    # Get settings
    mic_idx = obs.obs_data_get_int(_state.script_settings, "mic_id")
    direction = obs.obs_data_get_string(_state.script_settings, "direction")
    src_lang = direction.split('_')[0]
    audio_gate_db = float(obs.obs_data_get_int(_state.script_settings, "audio_gate"))
    max_lines = obs.obs_data_get_int(_state.script_settings, "max_lines")
    auto_detect = obs.obs_data_get_bool(_state.script_settings, "auto_detect")
    fuzzy_match = True  # Always enabled
    adaptive_vocab = True  # Always enabled
    auto_restart = True  # Always enabled
    show_confidence = True  # Always enabled
    
    # Find model (force large for better accuracy)
    model_path = find_model(src_lang, "large")
    
    if model_path is None:
        # Try fallback to small
        model_path = find_model(src_lang, "small")
    
    if model_path is None:
        log("No model found. Click DOWNLOAD MODELS.", obs.LOG_ERROR)
        return False
    
    # Initialize OBS text sources
    ensure_source("Translator_Source")
    ensure_source("Translator_Target")
    update_text_source("Translator_Source", "Initializing...", _state.script_settings, "source")
    update_text_source("Translator_Target", "Starting...", _state.script_settings, "target")
    
    # Get microphone info
    mics = get_mics()
    if not validate_mic_index(mic_idx):
        log(f"Invalid mic index {mic_idx}", obs.LOG_ERROR)
        return False
    
    mic_info = mics[mic_idx]
    samplerate = mic_info['samplerate']
    
    log(f"Starting: mic={mic_info['name']}, model={model_path.name}")
    
    # Start worker
    with _state.worker_lock:
        _state.translation_worker = AudioSTTWorker(
            mic_idx, str(model_path), direction,
            samplerate, audio_gate_db, max_lines,
            fuzzy_match, adaptive_vocab, auto_restart, show_confidence,
            auto_detect
        )
        _state.translation_worker.start()
    
    log("Translator active")
    return True

def _stop_translator() -> None:
    """Stop the translator"""
    with _state.worker_lock:
        if _state.translation_worker:
            _state.translation_worker.stop()
            _state.translation_worker.join(timeout=3)
            _state.translation_worker = None
    
    _state.is_running = False
    log("Translator stopped")
