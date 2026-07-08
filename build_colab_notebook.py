import json
import os

def main():
    notebook_path = "PropRAG_Benchmark_Colab.ipynb"
    print("Building self-contained Colab notebook...")

    # List of all benchmark files to bundle
    files_to_embed = [
        "benchmark/__init__.py",
        "benchmark/_bootstrap.py",
        "benchmark/bench_config.py",
        "benchmark/dataset.py",
        "benchmark/evaluation.py",
        "benchmark/llm_wrap.py",
        "benchmark/proprag_adapter.py",
        "benchmark/qa.py",
        "benchmark/report.py",
        "benchmark/results.py",
        "benchmark/run.py",
        "benchmark/smoke.py",
        "benchmark/usage.py",
        "benchmark/graphrag/__init__.py",
        "benchmark/graphrag/communities.py",
        "benchmark/graphrag/extract.py",
        "benchmark/graphrag/index.py",
        "benchmark/graphrag/prompts.py",
        "benchmark/graphrag/search.py"
    ]

    # Read each file
    files_dict = {}
    for rel_path in files_to_embed:
        if os.path.isfile(rel_path):
            with open(rel_path, "r", encoding="utf-8") as f:
                files_dict[rel_path] = f.read()
            print(f"  Loaded {rel_path} ({len(files_dict[rel_path])} chars)")
        else:
            print(f"  WARNING: {rel_path} not found!")

    # Serialize files_dict to a clean JSON string for embedding
    serialized_files = json.dumps(files_dict, ensure_ascii=False)

    # Let's build the notebook structure
    cells = []

    # Cell 1: Title
    cells.append({
        "cell_type": "markdown",
        "id": "title-markdown",
        "metadata": {},
        "source": [
            "# PropRAG vs GraphRAG vs BaseRAG — 2WikiMultiHopQA Benchmark\n",
            "\n",
            "Self-contained benchmark notebook for **Google Colab** (free T4 GPU).\n",
            "\n",
            "| Component | Details |\n",
            "|---|---|\n",
            "| **LLM** | `openai/gpt-oss-20b` loaded in **4-bit NF4** (bitsandbytes) — no GGUF/Koboldcpp needed |\n",
            "| **Embedder** | `BAAI/bge-large-en-v1.5` via sentence-transformers |\n",
            "| **Dataset** | 2WikiMultiHopQA (PropRAG-main repository clone or HuggingFace Hub) |\n",
            "\n",
            "### Quick-start\n",
            "1. Set `HF_TOKEN` in the **Configuration** cell (Cell 5).\n",
            "2. Go to `Runtime → Run all`.\n",
            "3. Results appear inline at the bottom and are saved to `/content/data/benchmark/`.\n",
            "\n",
            "> **Tip:** For a fast timing pilot, set `PILOT = 5`."
        ]
    })

    # Cell 2: Section 0 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec0-markdown",
        "metadata": {},
        "source": [
            "## Section 0 — GPU Check"
        ]
    })

    # Cell 3: GPU Check Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "gpu-check-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, sys\n",
            "\n",
            "r = subprocess.run([\"nvidia-smi\", \"--query-gpu=name,memory.total\", \"--format=csv,noheader\"],\n",
            "                   capture_output=True, text=True)\n",
            "if r.returncode == 0:\n",
            "    info = r.stdout.strip()\n",
            "    print(f\"GPU detected: {info}\")\n",
            "    vram = int(info.split(\",\")[1].strip().split()[0])\n",
            "    if vram < 14000:\n",
            "        print(f\"WARNING: only {vram} MB VRAM — gpt-oss-20b 4-bit needs ~14 GB. Consider Colab Pro (A100).\")\n",
            "    else:\n",
            "        print(f\"OK: {vram} MB VRAM — sufficient for 4-bit gpt-oss-20b.\")\n",
            "else:\n",
            "    print(\"No GPU found. Go to Runtime -> Change runtime type -> GPU (T4).\")"
        ]
    })

    # Cell 4: Section 1 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec1-markdown",
        "metadata": {},
        "source": [
            "## Section 1 — Install Dependencies\n",
            "> First run takes ~5 minutes. Subsequent runs skip already-installed packages."
        ]
    })

    # Cell 5: Install Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "install-deps-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "%%capture install_output\n",
            "# Core ML stack\n",
            "!pip install -q \"transformers>=4.48.0\" accelerate \"bitsandbytes>=0.43.0\" sentencepiece protobuf\n",
            "\n",
            "# Benchmark dependencies\n",
            "!pip install -q \"openai>=1.30\" \"sentence-transformers>=2.7\" \"python-igraph>=0.11\" \\\n",
            "               numpy pandas pyarrow \"tqdm>=4.66\" matplotlib huggingface_hub datasets\n",
            "\n",
            "# Lightweight OpenAI-compatible API server\n",
            "!pip install -q fastapi \"uvicorn[standard]\" pydantic\n",
            "\n",
            "print(\"All packages installed successfully.\")"
        ]
    })

    # Cell 6: Clone Repo Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "clone-repo-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "PROPRAG_MAIN = \"/content/PropRAG_main\"\n",
            "\n",
            "if not os.path.isdir(PROPRAG_MAIN):\n",
            "    print(\"Cloning PropRAG-main repository from GitHub...\")\n",
            "    ret = os.system(f\"git clone --quiet https://github.com/ReLink-Inc/PropRAG.git {PROPRAG_MAIN}\")\n",
            "    if ret != 0:\n",
            "        raise RuntimeError(\"git clone failed — check network connectivity.\")\n",
            "    print(f\"Cloned to {PROPRAG_MAIN}\")\n",
            "else:\n",
            "    print(f\"Already present: {PROPRAG_MAIN}\")\n",
            "\n",
            "# Confirm proprag_poc module is present\n",
            "poc_init = os.path.join(PROPRAG_MAIN, \"proprag_poc\", \"config.py\")\n",
            "if not os.path.isfile(poc_init):\n",
            "    raise ImportError(\n",
            "        f\"proprag_poc/config.py not found inside {PROPRAG_MAIN}.\\n\"\n",
            "        \"The GitHub repository layout may have changed — inspect the clone and adjust PROPRAG_MAIN.\"\n",
            "    )\n",
            "print(f\"proprag_poc found: {PROPRAG_MAIN}/proprag_poc/\")"
        ]
    })

    # Cell 7: Section 2 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec2-markdown",
        "metadata": {},
        "source": [
            "## Section 2 — Configuration\n",
            "> **Edit this cell to configure the evaluation run.**"
        ]
    })

    # Cell 8: Configuration Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "config-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── USER CONFIGURATION ────────────────────────────────────────────────────────\n",
            "\n",
            "# HuggingFace access token.\n",
            "# Required for openai/gpt-oss-20b (you must accept the model license first:\n",
            "# https://huggingface.co/openai/gpt-oss-20b)\n",
            "HF_TOKEN = \"\"   # e.g. \"hf_xxxxxxxxxxxxxxxxxxxx\"\n",
            "\n",
            "# Model to load\n",
            "MODEL_ID = \"openai/gpt-oss-20b\"\n",
            "\n",
            "# Embedding model (drop to BAAI/bge-base-en-v1.5 to save ~900 MB RAM)\n",
            "EMBEDDING_MODEL = \"BAAI/bge-large-en-v1.5\"\n",
            "\n",
            "# ── Benchmark knobs ───────────────────────────────────────────────────────────\n",
            "N_QUESTIONS   = 50        # Total questions to evaluate\n",
            "SEED          = 42\n",
            "PILOT         = None      # Set to e.g. 5 for a quick pilot; None for the full run\n",
            "SYSTEMS       = [\"BaseRAG\", \"GraphRAG\", \"PropRAG\"]\n",
            "FORCE_REINDEX = False     # True = rebuild all indexes even if cached\n",
            "MAKE_CHARTS   = True\n",
            "\n",
            "# ── Model loading ─────────────────────────────────────────────────────────────\n",
            "BNB_QUANT_TYPE         = \"nf4\"       # \"nf4\" or \"fp4\"\n",
            "BNB_COMPUTE_DTYPE      = \"float16\"   # \"float16\" or \"bfloat16\"\n",
            "BNB_DOUBLE_QUANT       = True\n",
            "MAX_NEW_TOKENS_DEFAULT = 1500        # Hard cap applied server-side\n",
            "\n",
            "# ── API server (no need to change) ───────────────────────────────────────────\n",
            "LLAMA_HOST = \"127.0.0.1\"\n",
            "LLAMA_PORT = 5001\n",
            "\n",
            "print(\"Configuration loaded.\")"
        ]
    })

    # Cell 9: Section 3 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec3-markdown",
        "metadata": {},
        "source": [
            "## Section 3 — HuggingFace Authentication & Path Setup"
        ]
    })

    # Cell 10: Auth Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "hf-auth-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys\n",
            "\n",
            "# HuggingFace token\n",
            "if HF_TOKEN:\n",
            "    os.environ[\"HF_TOKEN\"]               = HF_TOKEN\n",
            "    os.environ[\"HUGGING_FACE_HUB_TOKEN\"] = HF_TOKEN\n",
            "    try:\n",
            "        from huggingface_hub import login\n",
            "        login(token=HF_TOKEN, add_to_git_credential=False)\n",
            "        print(\"Logged in to HuggingFace Hub.\")\n",
            "    except Exception as e:\n",
            "        print(f\"HF login warning (non-fatal): {e}\")\n",
            "else:\n",
            "    print(\"No HF_TOKEN set. Download may fail if gpt-oss-20b is gated.\")\n",
            "\n",
            "# sys.path: proprag_poc lives inside PROPRAG_MAIN\n",
            "for p in [PROPRAG_MAIN, \"/content\"]:\n",
            "    if p not in sys.path:\n",
            "        sys.path.insert(0, p)\n",
            "\n",
            "# Env vars read by proprag_poc's POCConfig\n",
            "os.environ[\"PROPRAG_LLM_BACKEND\"]     = \"koboldcpp\"   # koboldcpp -> localhost:PORT/v1\n",
            "os.environ[\"PROPRAG_LLM_MODEL\"]       = MODEL_ID\n",
            "os.environ[\"PROPRAG_EMBEDDING_MODEL\"] = EMBEDDING_MODEL\n",
            "\n",
            "print(\"sys.path and env vars configured.\")"
        ]
    })

    # Cell 11: Section 4 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec4-markdown",
        "metadata": {},
        "source": [
            "## Section 4 — Write Benchmark Package to Colab Disk\n",
            "> Writes the benchmark source files to `/content/benchmark/`."
        ]
    })

    # Cell 12: Write Package Code (using robust json.loads to prevent escaping issues!)
    write_source = [
        "import json, os\n",
        "\n",
        "# Create directory structure\n",
        "os.makedirs(\"/content/benchmark/graphrag\", exist_ok=True)\n",
        "\n",
        "# Load and write all bundled benchmark files\n",
        f"serialized_files = {repr(serialized_files)}\n",
        "files_dict = json.loads(serialized_files)\n",
        "\n",
        "for path, content in files_dict.items():\n",
        "    full_path = \"/content/\" + path\n",
        "    os.makedirs(os.path.dirname(full_path), exist_ok=True)\n",
        "    with open(full_path, \"w\", encoding=\"utf-8\") as f:\n",
        "        f.write(content)\n",
        "    print(f\"Wrote {full_path} ({len(content)} chars)\")\n",
        "print(\"All benchmark files written successfully to Colab disk!\")"
    ]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "write-pkg-code",
        "metadata": {},
        "outputs": [],
        "source": write_source
    })

    # Cell 13: Section 5 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec5-markdown",
        "metadata": {},
        "source": [
            "## Section 5 — Download Dataset\n",
            "> Checks if the dataset files exist locally (e.g. from git clone) and copies them, otherwise downloads from HuggingFace Datasets."
        ]
    })

    # Cell 14: Download Dataset Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "download-ds-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, json, shutil\n",
            "\n",
            "DATA_DIR    = \"/content/data\"\n",
            "os.makedirs(DATA_DIR, exist_ok=True)\n",
            "\n",
            "REPO_DS_DIR  = os.path.join(PROPRAG_MAIN, \"reproduce\", \"dataset\")\n",
            "DATASET_PATH = os.path.join(DATA_DIR, \"2wikimultihopqa.json\")\n",
            "CORPUS_PATH  = os.path.join(DATA_DIR, \"2wikimultihopqa_corpus.json\")\n",
            "\n",
            "def _try_copy(basename, dst):\n",
            "    src = os.path.join(REPO_DS_DIR, basename)\n",
            "    if os.path.isfile(dst):\n",
            "        print(f\"  Already present: {dst}\")\n",
            "        return True\n",
            "    if os.path.isfile(src):\n",
            "        shutil.copy2(src, dst)\n",
            "        print(f\"  Copied from repository: {src}\")\n",
            "        return True\n",
            "    return False\n",
            "\n",
            "print(\"Locating 2WikiMultiHopQA dataset files...\")\n",
            "have_q = _try_copy(\"2wikimultihopqa.json\",        DATASET_PATH)\n",
            "have_c = _try_copy(\"2wikimultihopqa_corpus.json\",  CORPUS_PATH)\n",
            "\n",
            "if not (have_q and have_c):\n",
            "    print(\"Downloading from HuggingFace Datasets (voidful/2WikiMultiHopQA)...\")\n",
            "    from datasets import load_dataset\n",
            "    ds = load_dataset(\"voidful/2WikiMultiHopQA\", trust_remote_code=True)\n",
            "\n",
            "    if not have_q:\n",
            "        split = ds.get(\"validation\") or ds.get(\"train\")\n",
            "        records = []\n",
            "        for ex in split:\n",
            "            records.append({\n",
            "                \"_id\":             ex.get(\"id\", ex.get(\"_id\", str(len(records)))),\n",
            "                \"type\":            ex.get(\"type\", \"compositional\"),\n",
            "                \"question\":        ex[\"question\"],\n",
            "                \"answer\":          ex[\"answer\"],\n",
            "                \"supporting_facts\": [[sf[\"title\"], sf.get(\"sent_id\", 0)]\n",
            "                                     if isinstance(sf, dict) else sf\n",
            "                                     for sf in ex.get(\"supporting_facts\", [])],\n",
            "                \"context\": [[c[\"title\"], c.get(\"sentences\", [])]\n",
            "                            if isinstance(c, dict) else c\n",
            "                            for c in ex.get(\"context\", [])],\n",
            "            })\n",
            "        with open(DATASET_PATH, \"w\", encoding=\"utf-8\") as f:\n",
            "            json.dump(records, f)\n",
            "        print(f\"  Dataset saved: {len(records)} questions\")\n",
            "\n",
            "    if not have_c:\n",
            "        seen, corpus = set(), []\n",
            "        for sname in ds:\n",
            "            for ex in ds[sname]:\n",
            "                for ctx in ex.get(\"context\", []):\n",
                    "                    title = ctx[\"title\"] if isinstance(ctx, dict) else ctx[0]\n",
                    "                    sents = (ctx.get(\"sentences\", []) if isinstance(ctx, dict)\n",
                    "                             else (ctx[1] if len(ctx) > 1 else []))\n",
                    "                    if title not in seen:\n",
                    "                        seen.add(title)\n",
                    "                        corpus.append({\"title\": title, \"text\": \" \".join(sents) if sents else \"\"})\n",
            "        with open(CORPUS_PATH, \"w\", encoding=\"utf-8\") as f:\n",
            "            json.dump(corpus, f)\n",
            "        print(f\"  Corpus saved: {len(corpus)} documents\")\n",
            "\n",
            "os.environ[\"BENCH_DATASET_PATH\"] = DATASET_PATH\n",
            "os.environ[\"BENCH_CORPUS_PATH\"]  = CORPUS_PATH\n",
            "print(f\"\\nDataset path : {DATASET_PATH}\")\n",
            "print(f\"Corpus path  : {CORPUS_PATH}\")"
        ]
    })

    # Cell 15: Section 6 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec6-markdown",
        "metadata": {},
        "source": [
            "## Section 6 — Load `openai/gpt-oss-20b` with 4-bit NF4 Quantization\n",
            "> Downloads weights (~20 GB on first run — takes 10-20 min).\n",
            "> **Ensure you have accepted the model license** at https://huggingface.co/openai/gpt-oss-20b first."
        ]
    })

    # Cell 16: Load Model Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "load-model-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
            "\n",
            "_hf_kw = {\"token\": HF_TOKEN} if HF_TOKEN else {}\n",
            "\n",
            "print(f\"Loading tokenizer for {MODEL_ID}...\")\n",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **_hf_kw)\n",
            "print(\"Tokenizer loaded successfully.\")\n",
            "\n",
            "_compute_dtype = torch.float16 if BNB_COMPUTE_DTYPE == \"float16\" else torch.bfloat16\n",
            "bnb_config = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_quant_type=BNB_QUANT_TYPE,\n",
            "    bnb_4bit_compute_dtype=_compute_dtype,\n",
            "    bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,\n",
            ")\n",
            "\n",
            "print(f\"\\nLoading {MODEL_ID} in 4-bit NF4 (bitsandbytes)...\")\n",
            "print(\"  Downloading ~20 GB from HuggingFace. Please wait.\")\n",
            "\n",
            "try:\n",
            "    model = AutoModelForCausalLM.from_pretrained(\n",
            "        MODEL_ID,\n",
            "        quantization_config=bnb_config,\n",
            "        device_map=\"auto\",\n",
            "        attn_implementation=\"eager\",  # Workaround: gpt-oss uses custom attention layers\n",
            "        trust_remote_code=True,\n",
            "        **_hf_kw,\n",
            "    )\n",
            "    model.eval()\n",
            "    print(\"\\n✅ Model loaded successfully!\")\n",
            "    if torch.cuda.is_available():\n",
            "        used  = torch.cuda.memory_allocated() / 1e9\n",
            "        total = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
            "        print(f\"  GPU VRAM: {used:.1f} GB / {total:.1f} GB\")\n",
            "except Exception as e:\n",
            "    print(f\"\\n❌ Model loading failed: {e}\")\n",
            "    print(\"\\nTroubleshooting:\")\n",
            "    print(\"  1. Go to https://huggingface.co/openai/gpt-oss-20b and make sure you accepted the license terms.\")\n",
            "    print(\"  2. Paste your HuggingFace Token in Cell 3 (Configuration).\")\n",
            "    raise"
        ]
    })

    # Cell 17: Section 7 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec7-markdown",
        "metadata": {},
        "source": [
            "## Section 7 — Start OpenAI-Compatible API Server\n",
            "> Runs a FastAPI server in a background thread on `http://127.0.0.1:5001/v1`. Keep special tokens in model output so that Harmony reasoning stripping works."
        ]
    })

    # Cell 18: API Server Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "api-server-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import threading, time, uuid, asyncio\n",
            "import torch\n",
            "import uvicorn\n",
            "from fastapi import FastAPI\n",
            "from pydantic import BaseModel\n",
            "from typing import List, Optional, Any\n",
            "\n",
            "app = FastAPI(title=\"gpt-oss-20b API server\")\n",
            "\n",
            "class _Msg(BaseModel):\n",
            "    role: str\n",
            "    content: str\n",
            "\n",
            "class _ChatReq(BaseModel):\n",
            "    model:           str\n",
            "    messages:        List[_Msg]\n",
            "    max_tokens:      Optional[int]   = 512\n",
            "    temperature:     Optional[float] = 0.0\n",
            "    stream:          Optional[bool]  = False\n",
            "    response_format: Optional[Any]   = None\n",
            "    seed:            Optional[int]   = None\n",
            "\n",
            "@app.get(\"/v1/models\")\n",
            "def list_models():\n",
            "    return {\"object\": \"list\", \"data\": [\n",
            "        {\"id\": MODEL_ID, \"object\": \"model\", \"created\": 0, \"owned_by\": \"openai\"}\n",
            "    ]}\n",
            "\n",
            "@app.post(\"/v1/chat/completions\")\n",
            "async def chat_completions(req: _ChatReq):\n",
            "    messages    = [{\"role\": m.role, \"content\": m.content} for m in req.messages]\n",
            "    max_new     = min(req.max_tokens or 512, MAX_NEW_TOKENS_DEFAULT)\n",
            "    do_sample   = (req.temperature or 0.0) > 0.01\n",
            "    temperature = req.temperature if do_sample else 1.0\n",
            "\n",
            "    with torch.inference_mode():\n",
            "        try:\n",
            "            text = tokenizer.apply_chat_template(\n",
            "                messages, tokenize=False, add_generation_prompt=True,\n",
            "                reasoning_effort=\"medium\",\n",
            "            )\n",
            "        except Exception:\n",
            "            text = tokenizer.apply_chat_template(\n",
            "                messages, tokenize=False, add_generation_prompt=True,\n",
            "            )\n",
            "\n",
            "        inputs = tokenizer(\n",
            "            text, return_tensors=\"pt\",\n",
            "            truncation=True, max_length=6144,\n",
            "        ).to(model.device)\n",
            "\n",
            "        output_ids = model.generate(\n",
            "            **inputs,\n",
            "            max_new_tokens=max_new,\n",
            "            do_sample=do_sample,\n",
            "            temperature=temperature,\n",
            "            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id or 0,\n",
            "            eos_token_id=tokenizer.eos_token_id,\n",
            "        )\n",
            "\n",
            "    n_prompt = inputs[\"input_ids\"].shape[1]\n",
            "    new_ids  = output_ids[0][n_prompt:]\n",
            "    raw_text = tokenizer.decode(new_ids, skip_special_tokens=False,\n",
            "                                clean_up_tokenization_spaces=True)\n",
            "\n",
            "    return {\n",
            "        \"id\":      f\"chatcmpl-{uuid.uuid4().hex[:8]}\",\n",
            "        \"object\":  \"chat.completion\",\n",
            "        \"created\": int(time.time()),\n",
            "        \"model\":   MODEL_ID,\n",
            "        \"choices\": [{\"index\": 0,\n",
            "                     \"message\": {\"role\": \"assistant\", \"content\": raw_text},\n",
            "                     \"finish_reason\": \"stop\"}],\n",
            "        \"usage\":   {\"prompt_tokens\":     n_prompt,\n",
            "                    \"completion_tokens\": int(len(new_ids)),\n",
            "                    \"total_tokens\":      n_prompt + int(len(new_ids))},\n",
            "    }\n",
            "\n",
            "# Start uvicorn\n",
            "_cfg    = uvicorn.Config(app, host=LLAMA_HOST, port=LLAMA_PORT,\n",
            "                         log_level=\"warning\", loop=\"asyncio\")\n",
            "_server = uvicorn.Server(_cfg)\n",
            "_thread = threading.Thread(target=_server.run, daemon=True)\n",
            "_thread.start()\n",
            "\n",
            "# Wait up to 15 seconds\n",
            "import urllib.request, urllib.error\n",
            "_url = f\"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/models\"\n",
            "for _i in range(30):\n",
            "    time.sleep(0.5)\n",
            "    try:\n",
            "        with urllib.request.urlopen(_url, timeout=3): break\n",
            "    except Exception: pass\n",
            "else:\n",
            "    raise RuntimeError(\"FastAPI server did not start within 15 seconds.\")\n",
            "\n",
            "print(f\"✅ OpenAI-compatible server running at http://{LLAMA_HOST}:{LLAMA_PORT}/v1\")"
        ]
    })

    # Cell 19: Section 8 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec8-markdown",
        "metadata": {},
        "source": [
            "## Section 8 — Verify End-to-End (Smoke Test)"
        ]
    })

    # Cell 20: Smoke Test Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "smoke-test-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "from openai import OpenAI\n",
            "from benchmark.llm_wrap import strip_gpt_oss_reasoning\n",
            "\n",
            "client = OpenAI(\n",
            "    base_url=f\"http://{LLAMA_HOST}:{LLAMA_PORT}/v1\",\n",
            "    api_key=\"not-needed\",\n",
            ")\n",
            "\n",
            "resp = client.chat.completions.create(\n",
            "    model=MODEL_ID,\n",
            "    messages=[{\"role\": \"user\", \"content\": \"In one word, what is 2 + 2?\"}],\n",
            "    max_tokens=64,\n",
            "    temperature=0.0,\n",
            ")\n",
            "raw      = resp.choices[0].message.content\n",
            "stripped = strip_gpt_oss_reasoning(raw)\n",
            "print(f\"Raw output  : {raw!r}\")\n",
            "print(f\"After strip : {stripped!r}\")\n",
            "print(\"\\n✅ Smoke test passed!\")"
        ]
    })

    # Cell 21: Section 9 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec9-markdown",
        "metadata": {},
        "source": [
            "## Section 9 — Initialize Benchmark"
        ]
    })

    # Cell 22: Init Benchmark Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "init-bench-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys, os, logging, json\n",
            "\n",
            "if \"/content\" not in sys.path:\n",
            "    sys.path.insert(0, \"/content\")\n",
            "\n",
            "# Reload to make sure we load newly written files\n",
            "for mod in list(sys.modules):\n",
            "    if mod.startswith(\"benchmark\"):\n",
            "        del sys.modules[mod]\n",
            "\n",
            "from proprag_poc.logging_setup import setup_logging\n",
            "setup_logging()\n",
            "\n",
            "_handler = logging.StreamHandler(sys.stdout)\n",
            "_handler.setFormatter(logging.Formatter(\n",
            "    \"%(asctime)s | %(levelname)-5s | %(name)s | %(message)s\", \"%%H:%%M:%%S\"\n",
            "))\n",
            "bench_log = logging.getLogger(\"benchmark\")\n",
            "bench_log.setLevel(logging.INFO)\n",
            "bench_log.handlers.clear()\n",
            "bench_log.addHandler(_handler)\n",
            "bench_log.propagate = False\n",
            "\n",
            "from benchmark.bench_config         import BenchmarkConfig\n",
            "from benchmark.llm_wrap             import BenchLLMClient, check_backend\n",
            "from benchmark.dataset              import (build_corpus, corpus_id, load_questions,\n",
            "                                            pilot_subset, stratified_subset,\n",
            "                                            subset_hash, write_manifest, type_counts)\n",
            "from benchmark.evaluation           import em_score, f1_score, recall_at_k\n",
            "from benchmark.results              import ResultsStore\n",
            "from benchmark.usage               import IndexPhase, delta\n",
            "from benchmark                      import proprag_adapter as adapter\n",
            "from benchmark                      import report as report_mod\n",
            "from benchmark.graphrag             import index as gr_index_mod\n",
            "from benchmark.graphrag.search      import GraphRAGLocalRetriever\n",
            "from benchmark.qa                   import answer_question\n",
            "from proprag_poc.core.metrics       import get_usage_tracker\n",
            "from proprag_poc.embedding.encoder  import EmbeddingModel\n",
            "\n",
            "cfg = BenchmarkConfig(\n",
            "    project_dir  = \"/content\",\n",
            "    dataset_path = DATASET_PATH,\n",
            "    corpus_path  = CORPUS_PATH,\n",
            "    n_questions  = N_QUESTIONS,\n",
            "    seed         = SEED,\n",
            "    pilot        = PILOT,\n",
            ")\n",
            "poc_cfg = cfg.make_poc_config()\n",
            "systems = [s for s in SYSTEMS if s in (\"BaseRAG\", \"GraphRAG\", \"PropRAG\")]\n",
            "\n",
            "check_backend(poc_cfg)\n",
            "print(\"✅ Setup verified.\")"
        ]
    })

    # Cell 23: Prepare Dataset Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "prepare-ds-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import json, os\n",
            "\n",
            "all_qs    = load_questions(cfg.dataset_path)\n",
            "subset    = stratified_subset(all_qs, cfg.n_questions, cfg.seed)\n",
            "questions = pilot_subset(subset, cfg.pilot, cfg.seed) if cfg.pilot else subset\n",
            "\n",
            "print(f\"Loaded: {len(all_qs)} total | subset={len(subset)} | active={len(questions)}\")\n",
            "print(f\"Type counts: {type_counts(questions)}\")\n",
            "\n",
            "titles       = build_corpus(questions, cfg.corpus_path)\n",
            "docs         = [(t, titles[t]) for t in sorted(titles)]\n",
            "sub_hash     = subset_hash(questions)\n",
            "tag          = f\"n{cfg.n_questions}\" + (f\"_pilot{cfg.pilot}\" if cfg.pilot else \"\")\n",
            "corpus_ident = corpus_id(questions, tag)\n",
            "run_id       = f\"{cfg.n_questions}q_seed{cfg.seed}\" + (f\"_pilot{cfg.pilot}\" if cfg.pilot else \"\")\n",
            "run_dir      = os.path.join(cfg.data_dir, \"benchmark\", run_id)\n",
            "os.makedirs(run_dir, exist_ok=True)\n",
            "\n",
            "write_manifest(run_dir, questions, titles, cfg.seed, cfg)\n",
            "\n",
            "meta_path = os.path.join(run_dir, \"run_meta.json\")\n",
            "meta = {\n",
            "    \"subset_hash\": sub_hash, \"seed\": cfg.seed, \"n_questions\": cfg.n_questions,\n",
            "    \"pilot\": cfg.pilot, \"llm_backend\": poc_cfg.llm_backend,\n",
            "    \"llm_model\": poc_cfg.llm_model, \"embedding_model\": poc_cfg.embedding_model,\n",
            "    \"qa_top_k\": cfg.qa_top_k, \"retrieval_top_k\": cfg.retrieval_top_k,\n",
            "    \"gr_max_gleanings\": cfg.gr_max_gleanings,\n",
            "}\n",
            "if os.path.isfile(meta_path):\n",
            "    with open(meta_path) as f: prev = json.load(f)\n",
            "    if prev.get(\"subset_hash\") != sub_hash:\n",
            "        raise RuntimeError(\n",
            "            f\"Refusing to resume: existing run used subset {prev['subset_hash']!r}, \"\n",
            "            f\"current is {sub_hash!r}. Change seed/n_questions or delete {run_dir}.\"\n",
            "        )\n",
            "with open(meta_path, \"w\") as f: json.dump(meta, f, indent=2)\n",
            "\n",
            "print(f\"\\nCorpus ready: {len(docs)} documents | id={corpus_ident}\")\n",
            "print(f\"Run dir      : {run_dir}\")"
        ]
    })

    # Cell 24: Section 10 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec10-markdown",
        "metadata": {},
        "source": [
            "## Section 10 — Indexing Phase\n",
            "> Builds BaseRAG, PropRAG, and GraphRAG indexes. Cached and resume-safe."
        ]
    })

    # Cell 25: Indexing Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "indexing-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import time, json, os\n",
            "\n",
            "emb     = EmbeddingModel(poc_cfg)\n",
            "llm     = BenchLLMClient(poc_cfg)\n",
            "tracker = get_usage_tracker()\n",
            "cdir    = adapter.corpus_dir(poc_cfg, corpus_ident)\n",
            "usage_path  = os.path.join(run_dir, \"index_usage.json\")\n",
            "index_usage = {}\n",
            "\n",
            "bench_log.info(\"Indexing %d systems over %d docs (corpus %s)\",\n",
            "               len(systems), len(docs), corpus_ident)\n",
            "\n",
            "# ── BaseRAG ──────────────────────────────────────────────────────────────────\n",
            "bench_log.info(\"=== BaseRAG index ===\")\n",
            "with IndexPhase(tracker, \"BaseRAG\") as phase:\n",
            "    chunk_store, chunk_id_to_title = adapter.build_base_index(\n",
            "        poc_cfg, corpus_ident, docs, emb)\n",
            "index_usage[\"BaseRAG\"] = phase.usage\n",
            "bench_log.info(\"BaseRAG done: %.1f s\", phase.usage.get(\"wall_time_s\", 0))\n",
            "\n",
            "# ── PropRAG ──────────────────────────────────────────────────────────────────\n",
            "bench_log.info(\"=== PropRAG index ===\")\n",
            "with IndexPhase(tracker, \"PropRAG\") as phase:\n",
            "    corpus = adapter.build_or_load_proprag(\n",
            "        poc_cfg, corpus_ident, chunk_store, chunk_id_to_title,\n",
            "        emb, llm, force=FORCE_REINDEX)\n",
            "index_usage[\"PropRAG\"] = phase.usage\n",
            "bench_log.info(\"PropRAG done: %.1f s\", phase.usage.get(\"wall_time_s\", 0))\n",
            "\n",
            "# ── GraphRAG ─────────────────────────────────────────────────────────────────\n",
            "bench_log.info(\"=== GraphRAG index ===\")\n",
            "with IndexPhase(tracker, \"GraphRAG\", extra_scopes=[\"index::GraphRAG\"]) as phase:\n",
            "    gr_index = gr_index_mod.build_or_load(\n",
            "        poc_cfg, cfg, cdir, chunk_store, emb, llm, tracker,\n",
            "        force=FORCE_REINDEX)\n",
            "gr_usage = dict(phase.usage)\n",
            "gr_usage[\"parse_failures\"] = gr_index.n_extract_failures\n",
            "index_usage[\"GraphRAG\"] = gr_usage\n",
            "bench_log.info(\"GraphRAG done: %.1f s | %d entities | %d communities\",\n",
            "               gr_usage.get(\"wall_time_s\", 0),\n",
            "               len(gr_index.entities),\n",
            "               len(set(gr_index.communities.values())))\n",
            "\n",
            "with open(usage_path, \"w\") as f: json.dump(index_usage, f, indent=2)\n",
            "\n",
            "print(\"\\nAll indexes loaded:\")\n",
            "for s, u in index_usage.items():\n",
            "    print(f\"  {s:10s}: wall={u.get('wall_time_s',0):6.1f}s  \"\n",
            "          f\"prompt_tok={u.get('chat_prompt_tokens',0):6.0f}  \"\n",
            "          f\"compl_tok={u.get('chat_completion_tokens',0):6.0f}\")"
        ]
    })

    # Cell 26: Section 11 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec11-markdown",
        "metadata": {},
        "source": [
            "## Section 11 — Question Loop\n",
            "> Runs retrieval and generation for each system. Crash-safe and resume-safe."
        ]
    })

    # Cell 27: Question Loop Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "run-loop-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import time, json\n",
            "\n",
            "retrievers = {}\n",
            "if \"BaseRAG\"  in systems: retrievers[\"BaseRAG\"]  = adapter.make_baserag_retriever(corpus, emb, poc_cfg)\n",
            "if \"PropRAG\"  in systems: retrievers[\"PropRAG\"]  = adapter.make_proprag_retriever(corpus, emb, poc_cfg)\n",
            "if \"GraphRAG\" in systems: retrievers[\"GraphRAG\"] = GraphRAGLocalRetriever(gr_index, emb, poc_cfg, cfg)\n",
            "\n",
            "store = ResultsStore(run_dir)\n",
            "done  = store.done_keys()\n",
            "per_sys_secs = {s: [] for s in systems}\n",
            "\n",
            "t_start = time.monotonic()\n",
            "\n",
            "for qi, q in enumerate(questions, 1):\n",
            "    bench_log.info(\"Q %d/%d [%s] %s\", qi, len(questions), q.qtype, q.question[:80])\n",
            "    for system in systems:\n",
            "        if (q.qid, system) in done:\n",
            "            continue\n",
            "        retriever = retrievers[system]\n",
            "        scope     = f\"q::{system}\"\n",
            "        before    = tracker.snapshot(scope).as_dict()\n",
            "        try:\n",
            "            t0 = time.monotonic()\n",
            "            with tracker.scope(scope):\n",
            "                passages = retriever.retrieve(q.question, top_k=poc_cfg.retrieval_top_k)\n",
            "                ret_lat  = time.monotonic() - t0\n",
            "\n",
            "                ret_titles = [chunk_id_to_title.get(p.chunk_id, \"\") for p in passages]\n",
            "                recall     = recall_at_k(q.gold_titles, ret_titles, cfg.recall_ks)\n",
            "\n",
            "                if system == \"GraphRAG\":\n",
            "                    context = retriever.build_qa_context(q.question, passages)\n",
            "                else:\n",
            "                    context = [p.text for p in passages[:cfg.qa_top_k]]\n",
            "\n",
            "                t1 = time.monotonic()\n",
            "                answer, raw = answer_question(llm, q.question, context, cfg)\n",
            "                qa_lat = time.monotonic() - t1\n",
            "\n",
            "            usage = delta(before, tracker.snapshot(scope).as_dict())\n",
            "            row = {\n",
            "                \"qid\": q.qid, \"qtype\": q.qtype, \"system\": system,\n",
            "                \"question\": q.question, \"gold_answer\": q.answer,\n",
            "                \"gold_titles\": q.gold_titles, \"answer\": answer, \"raw_answer\": raw,\n",
            "                \"retrieved\": [{\"chunk_id\": p.chunk_id,\n",
            "                               \"title\": chunk_id_to_title.get(p.chunk_id, \"\"),\n",
            "                               \"score\": p.score} for p in passages],\n",
            "                \"recall\":              recall,\n",
            "                \"em\":                  em_score(q.gold_answers, answer),\n",
            "                \"f1\":                  f1_score(q.gold_answers, answer),\n",
            "                \"retrieval_latency_s\": round(ret_lat, 3),\n",
            "                \"qa_latency_s\":        round(qa_lat,  3),\n",
            "                \"usage\":               usage,\n",
            "                \"ts\":                  time.time(),\n",
            "            }\n",
            "            store.append(row)\n",
            "            per_sys_secs[system].append(ret_lat + qa_lat)\n",
            "            bench_log.info(\"  -> %s | em=%.0f f1=%.2f r@5=%.2f | %.1fs\",\n",
            "                           system, row[\"em\"], row[\"f1\"],\n",
            "                           recall.get(\"recall@5\", 0), ret_lat + qa_lat)\n",
            "        except Exception as _e:\n",
            "            bench_log.exception(\"Q %s system %s failed\", q.qid, system)\n",
            "            store.append({\"qid\": q.qid, \"qtype\": q.qtype, \"system\": system,\n",
            "                          \"error\": str(_e), \"ts\": time.time()})\n",
            "\n",
            "wall = time.monotonic() - t_start\n",
            "print(f\"\\nQuestion loop complete. Wall: {wall/60:.1f} min\")\n",
            "if cfg.pilot:\n",
            "    newly = [s for secs in per_sys_secs.values() for s in secs]\n",
            "    if newly:\n",
            "        mean_q = sum(newly) / len(newly)\n",
            "        proj   = mean_q * len(systems) * cfg.n_questions\n",
            "        print(f\"Pilot: mean {mean_q:.1f}s/call -> projected query loop wall ~{proj/60:.0f} min\")"
        ]
    })

    # Cell 28: Section 12 Title
    cells.append({
        "cell_type": "markdown",
        "id": "sec12-markdown",
        "metadata": {},
        "source": [
            "## Section 12 — Build Report & Display Results"
        ]
    })

    # Cell 29: Build Report Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "report-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "metrics     = report_mod.build(run_dir, make_charts=MAKE_CHARTS)\n",
            "report_path = os.path.join(run_dir, \"report.md\")\n",
            "print(f\"Report saved to: {report_path}\")"
        ]
    })

    # Cell 30: Display Report Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "display-report-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Markdown, display\n",
            "\n",
            "with open(report_path, \"r\", encoding=\"utf-8\") as f:\n",
            "    report_md = f.read()\n",
            "\n",
            "display(Markdown(report_md))"
        ]
    })

    # Cell 31: Display Charts Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "display-charts-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, matplotlib\n",
            "matplotlib.use(\"Agg\")\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.image as mpimg\n",
            "\n",
            "charts_dir = os.path.join(run_dir, \"charts\")\n",
            "if os.path.isdir(charts_dir):\n",
            "    chart_files = sorted(f for f in os.listdir(charts_dir) if f.endswith(\".png\"))\n",
            "    if chart_files:\n",
            "        fig, axes = plt.subplots(1, len(chart_files), figsize=(7 * len(chart_files), 5))\n",
            "        if len(chart_files) == 1:\n",
            "            axes = [axes]\n",
            "        for ax, fname in zip(axes, chart_files):\n",
            "            img = mpimg.imread(os.path.join(charts_dir, fname))\n",
            "            ax.imshow(img); ax.axis(\"off\")\n",
            "            ax.set_title(fname.replace(\".png\",\"\").replace(\"_\",\" \").title())\n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "    else:\n",
            "        print(\"No charts found.\")\n",
            "else:\n",
            "    print(\"Charts directory not found.\")"
        ]
    })

    # Cell 32: Summary Table Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "display-table-code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import json, os\n",
            "import pandas as pd\n",
            "from IPython.display import display\n",
            "\n",
            "metrics_path = os.path.join(run_dir, \"metrics.json\")\n",
            "if os.path.isfile(metrics_path):\n",
            "    with open(metrics_path) as f: m = json.load(f)\n",
            "    rows = []\n",
            "    for sys_name in [\"BaseRAG\", \"GraphRAG\", \"PropRAG\"]:\n",
            "        if sys_name not in m: continue\n",
            "        s = m[sys_name]\n",
            "        rows.append({\n",
            "            \"System\":         sys_name,\n",
            "            \"EM\":             f\"{s['em']*100:.1f}%\",\n",
            "            \"F1\":             f\"{s['f1']*100:.1f}%\",\n",
            "            \"Recall@5\":       f\"{s.get('recall@5',0)*100:.1f}%\",\n",
            "            \"Recall@10\":      f\"{s.get('recall@10',0)*100:.1f}%\",\n",
            "            \"Chat calls/Q\":   f\"{s['mean_chat_calls_per_q']:.2f}\",\n",
            "            \"QA latency (s)\": f\"{s['mean_qa_latency_s']:.2f}\",\n",
            "        })\n",
            "    df = pd.DataFrame(rows).set_index(\"System\")\n",
            "    display(df.style.set_caption(\"Summary Metrics Table\"))"
        ]
    })

    # Construct full notebook object
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Write notebook file cleanly
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"✅ Successfully wrote {notebook_path} ({len(cells)} cells)")

if __name__ == "__main__":
    main()
