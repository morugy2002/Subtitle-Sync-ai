# Third-party components

This project's own code is MIT-licensed (see `LICENSE`). It uses or downloads the following
components, each under its own license. Licenses listed here are to the best of the author's
knowledge - please check each project's page before redistributing.

| Component | Used for | License | Bundled in release ZIP? |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers) and its ONNX conversion by Xenova | Default matching model | Apache-2.0 | Yes (quantized ONNX) |
| LaBSE (Google) and its ONNX conversion by Xenova | Optional matching model | Apache-2.0 | No (downloaded on demand) |
| whisper.cpp (ggml-org) | Optional speech recognition | MIT | No (user installs) |
| OpenAI Whisper models (ggml conversions by ggerganov) | Optional speech recognition | MIT | No (downloaded on demand) |
| ffmpeg | Optional audio extraction | LGPL/GPL depending on build | No (user installs) |
| ONNX Runtime | Running the models | MIT | Yes |
| Hugging Face `tokenizers` | Text tokenization | Apache-2.0 | Yes |
| NumPy | Numerics | BSD-3-Clause | Yes |
| charset-normalizer | Encoding detection | MIT | Yes |
| Python / Tcl-Tk | Runtime and GUI | PSF / Tcl license | Yes |

Model sources:
- https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2
- https://huggingface.co/Xenova/LaBSE
- https://huggingface.co/ggerganov/whisper.cpp
