# 🐙 Octopus

Your own AI assistant: chat, coding, 3D modelling and images, with a Liquid Glass interface. It runs on **your** computer with no API and no subscription. Anything your computer can't handle runs on a **free cloud GPU** (Google Colab).

Octopus starts from free open-source models (Qwen 2.5 Coder for chat, Hunyuan3D-2 for 3D, SDXL or FLUX for images), and you make it yours by training it on your own examples.

![Octopus home](docs/home.png)

## Four kinds of chat

Pick one under **New chat** in the sidebar, or with the switcher at the top.

| | What it does |
|---|---|
| **Chat** | An everyday assistant with streaming replies, Markdown and tables |
| **Code** | Writes, explains and fixes code. Every code block has Copy, HTML gets a live **Preview**, and an optional **Run** button runs Python, JavaScript or Bash on your computer |
| **Modelling** | Turns a description or a picture into a **textured 3D model** (GLB, OBJ, STL) for Unity, Godot, Unreal, Roblox or Blender. You can spin it around, download it, and press **Use in Code** to get a playable Three.js scene with your model in it |
| **Image** | Makes pictures, concept art and icons. **Make 3D** turns any image into a model |

![Modelling](docs/modelling.png)

Plus **Train Octopus**: teach it with your own prompts and answers so it becomes *your* AI.

## What runs where

| Your computer | Chat & Code | Training, 3D & Images |
|---|---|---|
| **No NVIDIA GPU** (AMD, Intel, Mac, or a basic card) | On your computer through [Ollama](https://ollama.com/download), using your GPU or CPU | Free cloud notebook (Google Colab, NVIDIA T4). Results import into Octopus with one click |
| **NVIDIA GPU** | On your GPU | On your GPU (6 GB+ for 3D, 16 GB for textured "Quality" models) |

![Cloud notebook panel](docs/cloud.png)

## Install

You need **Python 3.10 to 3.12** from [python.org](https://www.python.org/downloads/). On Windows, tick **Add Python to PATH**.

**Without an NVIDIA GPU, install [Ollama](https://ollama.com/download) first** and open it once.

**Windows:** on GitHub press **Code → Download ZIP**, unzip it, and double-click **`start.bat`**.

**Mac or Linux:**

```bash
git clone https://github.com/physicstheory/octopus.git
cd octopus
./start.sh
```

Octopus opens at **http://localhost:8000**. The first chat downloads the model, about 1 GB for the default.

### Which chat model fits?

Pick it from the menu at the top of a chat.

| Model | Memory | Notes |
|---|---|---|
| Octopus Mini 0.5B | 1 to 2 GB | Very fast, simplest answers |
| **Octopus Base 1.5B** | 2 to 4 GB | Default, good balance |
| Octopus Plus 3B | 4 to 8 GB | Noticeably smarter |
| Octopus Pro 7B | 8 to 16 GB | Best coder here, slow without a strong GPU |

## Using the free cloud notebook

Open [**Octopus Cloud on Google Colab**](https://colab.research.google.com/github/physicstheory/octopus/blob/main/colab/Octopus_Cloud.ipynb) (also linked inside Octopus), sign in with Google, then choose **Runtime → Change runtime type → T4 GPU**.

- **Train a brain:** in Octopus open **Train Octopus**, tick your datasets and press **Download my data**. In the notebook, run cells 1 to 4 and upload that file when asked. Cell 4 downloads a `.gguf` brain. Back in Octopus press **Import brain**, and it appears in the model menu.
- **3D model:** run cells 1, 5 and 6. Type a description or upload a picture. It shows the model and downloads a `.zip`. In Octopus open **Modelling** and press **Import a 3D model**.
- **Images:** run cells 1 and 7.

Free Colab sessions end after a few hours or when idle, so download results when they finish. The notebook doesn't host a website, so it stays within Colab's free-tier rules.

## Training tips

- A few hundred good examples beat thousands of messy ones.
- Mix your own examples with an open dataset (Helpful chat, CodeAlpaca…) so Octopus keeps its general skills.
- If answers get weird or repetitive, train with **Quick** or a lower learning rate.
- Don't build training data by copying answers from ChatGPT, Claude, Gemini or similar. Their terms forbid using their outputs to train another AI model. Your own writing and openly licensed datasets are fine.

File formats for your examples:

```jsonl
{"prompt": "Who made you?", "response": "Duncan built me. I'm Octopus."}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey! What can I do for you?"}]}
```

CSV with `prompt,response` columns also works (or `instruction,output`, or `question,answer`), and so does plain text:

```text
User: What's the capital of France?
Assistant: Paris.
---
User: Write a haiku about octopuses
Assistant: Eight arms in the deep…
```

## 3D modelling details

- **Fast** uses Hunyuan3D-2 mini turbo, the quickest option at about 6 GB of VRAM. **Quality** uses Hunyuan3D-2, about 10 GB, or 16 GB with textures.
- **Game-ready** reduces the mesh to about 20,000 triangles. **Textures** paints colours and materials onto it.
- Text prompts first draw a reference picture (SDXL), then turn it into 3D. For the best results, describe one object on its own.
- Local 3D on an NVIDIA GPU: run `install_studio.bat` or `./install_studio.sh` once. Textures also need the CUDA Toolkit, plus Visual Studio Build Tools on Windows.
- **Licence:** Hunyuan3D-2 uses the Tencent Hunyuan 3D 2.0 Community License. It's free to use, including commercially, except in the EU, UK and South Korea, and products with over 1 million monthly users need a licence from Tencent. SDXL is CreativeML Open RAIL++-M and FLUX.1 schnell is Apache 2.0.

## Settings

Set these as environment variables before starting: `set NAME=value` on Windows, `export NAME=value` on Mac or Linux.

| Variable | Default | What it does |
|---|---|---|
| `OCTOPUS_ALLOW_RUN` | `0` | `1` turns on the **Run** button in Code chats. Code runs on your computer with your permissions, in a scratch folder, with a 30 second limit, and only from this computer. Read code before you run it. |
| `OCTOPUS_BACKEND` | `auto` | `ollama` or `transformers`. Auto uses transformers on NVIDIA and Ollama otherwise |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Where Ollama is listening |
| `OCTOPUS_PORT` | `8000` | Port for the web UI |
| `OCTOPUS_4BIT` | `0` | `1` loads chat models in 4-bit on NVIDIA (needs `pip install bitsandbytes`) |
| `OCTOPUS_HOST` | `127.0.0.1` | `0.0.0.0` lets you use Octopus from your phone on the same Wi-Fi (requires `OCTOPUS_PASSWORD`) |
| `OCTOPUS_PASSWORD` | none | Password for the sign-in screen. Without one, only this computer can use Octopus |
| `OCTOPUS_HOME` | `./workspace` | Where datasets, brains, images and 3D models are stored |

## How it works

```
web/                 Liquid Glass UI (plain HTML, CSS and JS, no build step)
octopus/
  server.py          FastAPI server: chat, studio, training, import and run APIs
  ollama_engine.py   Chat through Ollama (any GPU or CPU), GGUF brain import
  engine.py          Chat through transformers on NVIDIA, with LoRA brains
  trainer.py         LoRA fine-tuning with live progress
  studio.py          Images (diffusers) and image/text to 3D (Hunyuan3D-2)
  data.py            Reads JSONL/JSON/CSV/TXT and the open datasets
  runner.py          Runs code snippets for Code chats (opt-in)
colab/               The free cloud notebook
workspace/           Your datasets, brains and models (not committed to git)
```

`OCTOPUS_MOCK=1 python -m octopus` runs the interface with placeholder results, with no models needed.

## License

This code is MIT licensed. Each model keeps its own licence: Qwen 2.5 Coder is Apache 2.0, and Hunyuan3D-2, SDXL and FLUX are covered above. Each built-in dataset shows its licence in the Train section.
