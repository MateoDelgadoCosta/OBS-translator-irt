# OBS Real-Time Speech Translator v7.0

Real-time speech-to-text and translation (English <-> Spanish) plugin for OBS Studio.

---

## Why Manual Installation?

This plugin requires manual installation for **your security**:

- **No automatic installers**: Scripts `.bat` or `.exe` can be blocked by Smart App Control or antivirus
- **Safety first**: Manual installation prevents unauthorized code execution on your PC
- **Transparency**: You control exactly what gets installed
- **OBS Scripting**: Uses OBS's built-in Python scripting (Tools > Scripts), no external dependencies

---

## Features

- **Fast Whisper STT**: GPU-accelerated speech recognition using Faster-Whisper (Large-v3)
- **Vosk Fallback**: Offline Vosk recognition if GPU unavailable
- **Smart Engine Switching**: Automatically switches between GPU/CPU/Vosk based on VRAM and GPU utilization
- **High-Res Audio**: Supports up to 192kHz input with automatic downsampling to 16kHz
- **Hysteresis Audio Gate**: -50dB open / -65dB close to prevent phrase-chopping
- **Auto Translation**: Google Translate with caching (reduces API calls)
- **Thread-Safe UI**: All OBS API calls from main thread via queue + timer callback
- **Memory Safety**: Proper initialization in __init__, clean shutdown sequence
- **Chunk Processing**: Background processing during speech, only finalized text shown

## Installation

### Quick Start (Recommended)

1. Open OBS Studio
2. Go to **Tools > Scripts**
3. Click **+**, select `obs_translator_v7.py`
4. Click **"INSTALL DEPS"** button (installs Python packages)
5. Click **"DOWNLOAD MODELS"** button (downloads Whisper model)
6. Select your microphone and click **START**

### Requirements

- **Python**: 3.11+
- **GPU** (optional): NVIDIA with CUDA for fastest transcription
- **Dependencies**: faster-whisper, numpy, scipy, sounddevice, deep-translator, pynvml

### Manual Installation

**Step 1: Install Python**
1. Download Python 3.11+ from: https://www.python.org/downloads/
2. Run installer, **CHECK** "Add Python to PATH"
3. Click "Install Now"

**Step 2: Install Dependencies**
1. Open **Command Prompt** (Win+R, type `cmd`, Enter)
2. Run this command:
```
pip install faster-whisper numpy scipy sounddevice deep-translator pynvml
```

**Step 3: Install OBS Script**
1. Create folder: `%APPDATA%\obs-studio\obs-plugins\obs-scripting\python_scripts\`
2. Copy `obs_translator_v7.py` to that folder

**Step 4: Configure OBS**
1. Open OBS Studio
2. Go to **Tools > Scripts**
3. Click **+** button
4. Select `obs_translator_v7.py`
5. Configure microphone and language
6. Click **"DOWNLOAD MODELS"** button (downloads speech model)
7. Click **START**, wait until it says listening in the text on the screen

## Usage

### Setup Display Source

**Important**: Make sure to **select a font** in the script's dropdown on both before clicking START, otherwise text won't appear.

The translator automatically creates two text sources:
1. **Translator_Source** - Shows recognized speech
2. **Translator_Target** - Shows translation

Position these sources in your scene. The script updates them automatically.

**Tip**: Speak for several seconds to fill the text width, making it easier to position the sources in your scene.

### Configuration

| Setting | Description |
|---------|-------------|
| Microphone | Select your input device |
| Language | English→Spanish or Spanish→English |
| Engine | Auto (GPU→CPU→Vosk), Whisper GPU, Whisper CPU, or Vosk |
| Audio Gate | Minimum volume to detect speech (-50dB open, -65dB close) |
| Max Lines | Maximum lines to display (wraps after) |
| Custom Word | Add words to improve recognition |

### Engine Selection

The script automatically selects the best engine based on your GPU:
- **GPU Available**: Uses Faster-Whisper with CUDA (fastest)
- **No GPU / GPU Busy**: Falls back to Faster-Whisper CPU (int8 quantization)
- **Whisper Unavailable**: Falls back to Vosk (offline, slower but reliable)

GPU monitoring checks:
- VRAM available (>2GB required for GPU mode)
- GPU utilization (<80% to avoid lag)

### Auto-Detect Feature

**Important**: Enable **Auto-detect Language** for best results. This feature:
- Automatically detects if you speak Spanish or English
- If you speak in the **destination language**, shows text directly (no translation needed)
- If you speak in a **different language**, translates to destination

**Language Direction** must be set correctly for auto-detect to work:
- Set it to the language your audience expects
- Example: If you speak Spanish and your audience wants Spanish subtitles, set direction to **Spanish→English** (the destination language)

Examples:
- **ES → EN, speak Spanish**: Shows Spanish (Spanish = destination)
- **ES → EN, speak English**: Translates to Spanish
- **EN → ES, speak English**: Shows English (English = destination)
- **EN → ES, speak Spanish**: Translates to English

### Styling

- **Font**: Use clear fonts like Verdana, Roboto, Arial, Open Sans, or Helvetica (Bold recommended)
- **Font Size**: 24-100px range
- **Colors**: Custom text and background colors
- **Opacity**: Adjustable background transparency
- **Text Width**: Default 360px (25% less than previous version)

## Troubleshooting

### "No model found" Error
Verify models are in: `%APPDATA%\OBS_Translator\models\`
- Faster-Whisper: Models downloaded automatically

### Text Not Displaying
1. Check if script is running (click START)
2. Make sure Translator_Source and Translator_Target sources exist in scene
3. Try clicking STOP then START again

### GPU Not Being Used
1. Make sure NVIDIA GPU drivers are installed
2. Verify CUDA is available (`nvidia-smi`)
3. Check GPU utilization and VRAM

### Poor Recognition
1. Use a good quality microphone
2. Reduce background noise
3. Speak clearly and at normal pace

### Translation Not Working
Check your internet connection. Google Translate requires internet access.

### Crash on OBS Close
Fixed. Click STOP before closing OBS for clean shutdown.

## Files Included

| File | Description |
|------|-------------|
| `obs_translator_v7.py` | Main OBS script |
| `README.md` | This file |
| `obs_translator_v7_backup_UI_FIXED.py` | Backup of stable version |

## Data Location

Models and custom vocabulary are stored in:
- Windows: `%APPDATA%\OBS_Translator\`
- macOS: `~/Library/Application Support/OBS_Translator/`
- Linux: `~/.config/OBS_Translator/`

## License

MIT License - Free to use and modify.

## Version History

- **v7.0**: Faster-Whisper (Large-v3) with GPU support, Vosk fallback, thread-safe UI, hysteresis audio gate
- **v5.12**: Thread-safe logging - OBS API calls only from main thread
- **v5.11**: Translation cache + memory cleanup for long sessions
- **v5.10**: OBS API safety flags to prevent crashes
- **v5.9**: Auto language detection with fast-langdetect
- **v5.8**: Text centered horizontally and vertically
- **v5.7**: Text restart on overflow, only finalized text shown
- **v5.6**: Fixed phrase repetition, improved deduplication
- **v5.5**: Thread-safe stream management, clean shutdown
- **v5.4**: Fixed OBS close crash, improved shutdown sequence
- **v5.3**: Added custom vocabulary, improved text wrapping
- **v5.2**: Added audio processing filters
- **v5.1**: Thread-safe architecture
- **v5.0**: Initial release with Vosk STT
