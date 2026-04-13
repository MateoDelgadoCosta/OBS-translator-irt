# OBS Real-Time Speech Translator v5.12

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

- **Real-time STT**: Offline speech recognition using Vosk
- **Auto Translation**: Google Translate with caching (reduces API calls)
- **Auto Language Detection**: Automatically detects EN/ES with fast-langdetect
- **Translation Cache**: Caches translations to reduce API calls by ~70%
- **Memory Cleanup**: Automatic cleanup every 30 min for long sessions
- **Audio Processing**: Noise gate, compression, high-pass filter
- **Custom Vocabulary**: Add words to improve recognition
- **Auto-restart**: Automatic mic reconnection
- **Clean Display**: Text wrapping with line limits

## Installation

### Quick Start (Recommended)

1. Open OBS Studio
2. Go to **Tools > Scripts**
3. Click **+**, select `obs_translator_v5.py`
4. Click **"INSTALL DEPS"** button (installs Python packages)
5. Click **"DOWNLOAD MODELS"** button (downloads speech models)
6. Select your microphone and click **START**

### Manual Installation

**Step 1: Install Python**
1. Download Python 3.11+ from: https://www.python.org/downloads/
2. Run installer, **CHECK** "Add Python to PATH"
3. Click "Install Now"

**Step 2: Install Dependencies**
1. Open **Command Prompt** (Win+R, type `cmd`, Enter)
2. Run these commands one by one:
```
pip install numpy scipy sounddevice vosk deep_translator fast-langdetect
```

**Step 3: Download Speech Models**
1. Create folder: `%APPDATA%\OBS_Translator\models\`
2. Download English model:
   - Go to: https://alphacephei.com/vosk/models
   - Download: `vosk-model-en-us-0.22-lgraph` (~1.2GB)
   - Extract ZIP to: `%APPDATA%\OBS_Translator\models\`
   - Rename folder to: `vosk-model-en-us-0.22-lgraph`
3. Download Spanish model:
   - Download: `vosk-model-es-0.42` (~1.4GB)
   - Extract ZIP to: `%APPDATA%\OBS_Translator\models\`
   - Rename folder to: `vosk-model-es-0.42`

**Step 4: Install OBS Script**
1. Create folder: `%APPDATA%\obs-studio\obs-plugins\obs-scripting\python_scripts\`
2. Copy `obs_translator_v5.py` to that folder

**Step 5: Configure OBS**
1. Open OBS Studio
2. Go to **Tools > Scripts**
3. Click **+** button
4. Select `obs_translator_v5.py`
5. Configure microphone and language
6. Click **START**

## Usage

### Setup Display Source

**Important**: Make sure to **select a source** in the script's dropdown before clicking START, otherwise text won't appear.

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
| Auto-detect | Automatically detects spoken language |
| Audio Gate | Minimum volume to detect speech (-60 dB default) |
| Max Lines | Maximum lines to display (wraps after) |
| Custom Word | Add words to improve recognition |

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

## Troubleshooting

### "No model found" Error
Verify models are in: `%APPDATA%\OBS_Translator\models\`
- English: `vosk-model-en-us-0.22-lgraph\`
- Spanish: `vosk-model-spa-0.42\`

### Text Not Displaying
1. Check if script is running (click START)
2. Make sure Translator_Source and Translator_Target sources exist in scene
3. Try clicking STOP then START again

### Poor Recognition
1. Add custom words in the script settings
2. Use a good quality microphone
3. Reduce background noise
4. Speak clearly and at normal pace

### Translation Not Working
Check your internet connection. Google Translate requires internet access.

### Crash on OBS Close
Fixed in v5.5+. If still happening, click STOP before closing OBS.

## Files Included

| File | Description |
|------|-------------|
| `obs_translator_v5.py` | Main OBS script |
| `download_models.py` | Model downloader (optional) |
| `README.md` | This file |

## Data Location

Models and custom vocabulary are stored in:
- Windows: `%APPDATA%\OBS_Translator\`
- macOS: `~/Library/Application Support/OBS_Translator/`
- Linux: `~/.config/OBS_Translator/`

## License

MIT License - Free to use and modify.

## Version History

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
