# OBS Real-Time Speech Translator v5.7

Real-time speech-to-text and translation (English <-> Spanish) plugin for OBS Studio.

## Features

- **Real-time STT**: Offline speech recognition using Vosk
- **Auto Translation**: Google Translate integration
- **Audio Processing**: Noise gate, compression, high-pass filter
- **Custom Vocabulary**: Add words to improve recognition
- **Auto-restart**: Automatic mic reconnection
- **Clean Display**: Text wrapping with line limits

## Installation (Manual - No Scripts)

### Step 1: Install Python
1. Download Python 3.11+ from: https://www.python.org/downloads/
2. Run installer, **CHECK** "Add Python to PATH"
3. Click "Install Now"

### Step 2: Install Dependencies
1. Open **Command Prompt** (Win+R, type `cmd`, Enter)
2. Run these commands one by one:
```
pip install numpy
pip install scipy
pip install sounddevice
pip install vosk
pip install deep_translator
```

### Step 3: Install OBS Script
1. Create folder: `%APPDATA%\obs-studio\obs-plugins\obs-scripting\python_scripts\`
2. Copy `obs_translator_v5.py` to that folder

### Step 4: Configure OBS
1. Open OBS Studio
2. Go to **Tools > Scripts**
3. Click **+** button
4. Select `obs_translator_v5.py`
5. Configure microphone, language and fonts (72 recommended size)

### Step 5: Download Speech Models (see progress with script log button)


## Usage

### Setup Display Source

The translator automatically creates two text sources:
1. **Translator_Source** - Shows recognized speech
2. **Translator_Target** - Shows translation

Position these sources in your scene. The script updates them automatically.

### Configuration

| Setting | Description |
|---------|-------------|
| Microphone | Select your input device |
| Language | English→Spanish or Spanish→English |
| Audio Gate | Minimum volume to detect speech (-60 dB default) |
| Max Lines | Maximum lines to display (wraps after) |
| Custom Word | Add words to improve recognition |

### Styling

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
| `README.md` | This file |

## Data Location

Models and custom vocabulary are stored in:
- Windows: `%APPDATA%\OBS_Translator\`
- macOS: `~/Library/Application Support/OBS_Translator/`
- Linux: `~/.config/OBS_Translator/`

## License

MIT License - Free to use and modify.

## Version History

- **v5.7**: Text restart on overflow, only finalized text shown
- **v5.6**: Fixed phrase repetition, improved deduplication
- **v5.5**: Thread-safe stream management, clean shutdown
- **v5.4**: Fixed OBS close crash, improved shutdown sequence
- **v5.3**: Added custom vocabulary, improved text wrapping
- **v5.2**: Added audio processing filters
- **v5.1**: Thread-safe architecture
- **v5.0**: Initial release with Vosk STT
