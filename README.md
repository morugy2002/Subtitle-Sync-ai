**English** | [فارسی](README.fa.md)

# Subtitle Sync AI

Fix an out-of-sync subtitle (for example a Persian one) by matching it, **by meaning**, to a subtitle
that is already in sync (in any language), or to the movie's own audio. Works offline on Windows.

Inspired by [subsync](https://github.com/sc0ty/subsync). This project is a new implementation and
contains no code from it.

> **Made with AI.** This project was written with the help of **Claude** (an AI model by Anthropic,
> Claude Sonnet 5.5) in a conversation with the repository owner, who tested it on real subtitles.
> It has only been tested on a limited set of files, so please report any bug you find.

## Features

- Sync any subtitle to a reference subtitle in **any language pair** (MiniLM ~50 languages, LaBSE ~100).
- Handles a constant offset, speed drift (frame-rate mismatch) and cut/extra scenes.
- Reads SRT, VTT and ASS/SSA; writes SRT. Auto-detects the encoding (including old Windows-1256 Persian files).
- Persian text fixes (Arabic ي/ك → ی/ک, Arabic digits, tatweel) and an option for RTL punctuation problems.
- Bilingual output (reference text under each line) - handy for language learning.
- *Optional:* create a subtitle from a video in its original language, or translate speech to English
  (Whisper via whisper.cpp), or sync directly against the video's audio.
- No Python needed for the release build. Nothing is uploaded anywhere; everything runs on your PC.

## Download (Windows)

Get `SubtitleSyncAI.zip` from the [Releases](https://github.com/morugy2002/subtitle-sync-ai/releases)
page, extract it, and run `SubtitleSyncAI.exe`. The MiniLM model is included (~230 MB unpacked).

## Usage

1. **Sync subtitle** tab → choose the subtitle to fix, then the reference (an in-sync subtitle in any
   language, or the video), choose where to save, press *Synchronize*.
2. Check the result in your player at the beginning, middle and end of the movie.
3. If matching is poor, lower **Min similarity** (e.g. 0.35); if you see wrong matches, raise it.

### Optional components (Components tab)

| Feature | What to install |
|---|---|
| Higher-accuracy model (experimental) | Download **LaBSE** (~480 MB) |
| Subtitle from video / sync to audio | **whisper.cpp** (`whisper-bin-x64.zip`, or `whisper-cublas-…` for NVIDIA GPUs), **ffmpeg**, and a **Whisper model** - use *Install from ZIP* / *Download* |

## Run from source

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
python app.py
```

Use a fresh virtual environment (no PyTorch needed). Python 3.10-3.12 recommended.

## Build the exe

Double-click `build_exe.bat`. It creates a clean environment, downloads MiniLM, runs PyInstaller and
produces `SubtitleSyncAI.zip`.

## How it works

1. Both subtitles are turned into multilingual sentence embeddings (ONNX Runtime, no PyTorch).
2. A monotone dynamic-programming alignment pairs lines with the same meaning.
3. Outlier matches are removed; a second pass uses the time structure found in the first.
4. Timing is taken from the matched reference lines and interpolated for the rest.

## Known limitations

- The interface is in English (Tk does not render right-to-left text consistently on all systems).
- Very free translations, or subtitles from a different cut of the movie, may match poorly.
- LaBSE support is experimental. Whisper timing is less precise than a good subtitle's.
- Windows only is tested. Antivirus programs sometimes flag PyInstaller apps (false positive);
  you can verify the release checksum or build it yourself.

## Ideas / roadmap

Batch mode for a whole season · extracting subtitles from MKV · offline or API translation to Persian ·
a Persian interface. Issues and pull requests are welcome.

## License and credits

MIT - see [LICENSE](LICENSE). Third-party models and tools: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
