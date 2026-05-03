# Orchestra-o1: Omnimodal Agent Orchestration

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Framework-verl-orange.svg" alt="Framework: verl">
</p>

Orchestra-o1 is a **multi-agent orchestration framework** that decomposes complex omnimodal tasks into parallel subtasks, delegating them to specialized SubAgents for execution. It features a **MainAgent + SubAgent** architecture where the MainAgent (orchestrator) plans and coordinates, while SubAgents execute specific subtasks using various tools.

## 🏗️ Architecture

<p align="center">
  <img src="figs/orchestra-o1-framework.png" width="100%" alt="Orchestra-o1 Framework">
</p>

<p align="center"><b>Figure 1.</b> Overview of the Orchestra-o1 framework. The MainAgent orchestrates multi-turn interactions by decomposing omnimodal tasks into independent/dependent subtasks, creating specialized SubAgents with perception tools (image, audio, video analysis) and action tools (web search, page visit, code execution), and executing them in parallel. An online sub-agent specialization module handles sub-task preparation, model selection, tool integration, and memory allocation.</p>

## 📦 Model Weights

The trained Orchestra-o1-8B model weights are available at: [[🤗 Model](https://huggingface.co/Karl28/Orchestra-o1-8B)]

## ✨ Key Features

- **🎯 Hierarchical Multi-Agent Architecture**: MainAgent orchestrates task decomposition; SubAgents execute subtasks with specialized tools
- **⚡ Parallel Subtask Execution**: Independent subtasks run simultaneously, maximizing throughput
- **🔧 Rich Tool Ecosystem**: Web search, code execution, video/audio/image analysis, URL extraction
- **🧠 GRPO Training**: Train open-source models (Qwen3-8B) as MainAgent using Group Relative Policy Optimization with LLM-as-judge reward
- **📊 OmniGAIA Benchmark**: Comprehensive evaluation on omnimodal question-answering tasks

## 📁 Project Structure

```
Orchestra-o1/
├── bench_orchestra_o1_omnigaia.py   # 🚀 Inference with commercial models (e.g., GPT-5)
├── bench_qwen/                       # 🚀 Inference with trained open-source models
│   ├── run_qwen3_8b_grpo.sh         #    One-click: vLLM + benchmark (Qwen3-8B GRPO)
│   ├── bench_qwen_omnigaia.py        #    Benchmark runner for Qwen3-8B
│   ├── eval_qwen.py                  #    Evaluation report generator
│   ├── model_config_qwen.yaml        #    Model config (local vLLM + commercial APIs)
│   └── orchestra_o1_omnigaia_qwen_grpo.yaml  # Benchmark config
├── benchmark/                        # Benchmark framework
│   ├── common/                       #    Runner, environment abstractions
│   └── omnigaia/                     #    OmniGAIA benchmark implementation & tools
├── base/                             # Base framework
│   ├── agent/                        #    Agent abstractions (BaseAgent, Memory, ReAct)
│   └── engine/                       #    LLM engine, cost monitoring, logging
├── orchestra_o1/                     # 🎵 Core orchestration framework
│   ├── main_agent.py                 #    MainAgent (orchestrator)
│   ├── config.py                     #    Configuration loader
│   ├── prompts/                      #    Prompt templates (OmniGAIA)
│   ├── runners/                      #    Benchmark runners
│   ├── subagents/                    #    SubAgent implementations (ReAct)
│   └── tools/                        #    Orchestration tools (delegate, complete, trace)
├── train_qwen3_8b/                   # 🏋️ GRPO Training
│   ├── grpo/                         #    GRPO training pipeline
│   │   ├── train_grpo_qwen3_8b.sh   #    Training script (8×H20 GPUs)
│   │   └── reward_fn.py             #    LLM-as-judge multi-dimensional reward
│   └── ds_config.json               #    DeepSpeed config
├── config/                           # Configuration files
│   ├── model_config.yaml             #    LLM API configuration
│   └── benchmarks/                   #    Benchmark configurations
├── eval/                             # Evaluation scripts
├── .env.example                      # Environment variables template
├── requirements.txt                  # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/zfkarl/Orchestra-o1.git
cd Orchestra-o1

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (JINA_API_KEY, SERPER_API_KEY, OPENAI_API_KEY)
```

### 2. Configure Models

Edit `config/model_config.yaml` with your LLM API credentials:

```yaml
models:
  "gpt-5":
    api_type: "openai"
    base_url: "https://api.openai.com/v1/"
    api_key: "your_api_key_here"
```

### 3. Prepare Dataset

Download the [OmniGAIA](https://huggingface.co/datasets/RUC-NLPIR/OmniGAIA) dataset and place it under `data/OmniGAIA/`.

### 4. Run Inference

#### Mode A: Commercial Model (GPT-5) as MainAgent

```bash
python bench_orchestra_o1_omnigaia.py --config config/benchmarks/orchestra_o1_omnigaia.yaml
```

#### Mode B: Trained Open-Source Model (Qwen3-8B GRPO) as MainAgent

```bash
# One-click: starts vLLM server → waits for ready → runs benchmark → generates report
VLLM_MODEL_PATH=/path/to/your/grpo/checkpoint/huggingface \
CUDA_VISIBLE_DEVICES=0 \
bash bench_qwen/run_qwen3_8b_grpo.sh
```

## 🏋️ GRPO Training

Train Qwen3-8B as the MainAgent using GRPO (Group Relative Policy Optimization) with an LLM-as-judge reward function.

<p align="center">
  <img src="figs/orchestra-o1-8B-training.png" width="100%" alt="Orchestra-o1-8B Training Pipeline">
</p>

<p align="center"><b>Figure 2.</b> Orchestra-o1-8B training pipeline. <b>(a) Training Data Curation</b>: Starting from seed data, we run Orchestra-o1 (GPT-5) to collect trajectories, extract anchor facts across modalities, apply QA rewrites (pivot swapping, temporal shifting, etc.), and filter & verify to produce 1.2K high-quality training samples. <b>(b) DA-GRPO Training</b>: We reconstruct decision examples from expert trajectories, sample G candidate decisions from the base model (Qwen3-8B), score each on 4 dimensions via a rubric reward (format, action, tool, decision quality), compute relative advantages, and optimize with DA-GRPO to produce Orchestra-o1-8B.</p>

### Prerequisites

- **Hardware**: 8× H20 (96GB) GPUs (single node)
- **Software**: [verl](https://github.com/volcengine/verl) framework
- **Data**: GPT-5 expert trajectories from OmniGAIA benchmark

### Training Pipeline

```bash
cd train_qwen3_8b/grpo

# Launch GRPO training
VERL_DIR=/path/to/verl MODEL_PATH=/path/to/Qwen3-8B \
bash train_grpo_qwen3_8b.sh
```

### Reward Function

The reward function (`reward_fn.py`) uses **LLM-as-judge** (claude-haiku-4-5) to evaluate 4 dimensions:

| Dimension | Weight | Range | Description |
|---|---|---|---|
| **format_correct** | 0.10 | {0, 1} | JSON format correctness |
| **action_valid** | 0.10 | {0, 1} | Action validity (delegate_task / complete) |
| **tool_reasonable** | 0.20 | [0, 1] | Tool selection & subtask assignment quality |
| **decision_quality** | **0.60** ★ | [0, 1] | Overall decision quality (references GPT-5 expert) |

`score = 0.10 × format + 0.10 × action + 0.20 × tool + 0.60 × decision ∈ [0, 1]`

### MainAgent Decision Flow

1. **Analyze** the question and omnimodal inputs
2. **Decompose** into independent subtasks (Phase 1)
3. **Delegate** all independent subtasks in parallel
4. **Evaluate** results — sufficient? → complete; need more? → plan Phase 2
5. **Iterate** until answer is found or budget exhausted

### SubAgent (ReAct)

Each SubAgent follows the ReAct (Reasoning + Acting) paradigm:
- **Think**: Reason about the current state
- **Act**: Use tools (search, code execution, media analysis)
- **Observe**: Process tool outputs
- **Repeat** until task is complete

## 📊 Evaluation

After running the benchmark, generate evaluation reports:

```bash
# For commercial model results
python eval/eval.py --main_agent gpt-5

# For Qwen3-8B GRPO results
python bench_qwen/eval_qwen.py --csv_path logs/omnigaia_qwen_8b_grpo/omnigaia_qwen_xxx.csv --main_agent qwen3-8b-grpo
```

## 🔧 Configuration

### Environment Variables (`.env`)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (for GPT-5, GPT-4o, etc.) |
| `OPENAI_BASE_URL` | OpenAI API base URL |
| `JINA_API_KEY` | Jina API key (for web content extraction) |
| `SERPER_API_KEY` | Serper API key (for Google search) |

### Key Training Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `TRAIN_BSZ` | 24 | Training batch size |
| `ROLLOUT_N` | 8 | GRPO group size |
| `MAX_PROMPT_LEN` | 24576 | Max prompt length (tokens) |
| `MAX_RESP_LEN` | 4096 | Max response length (tokens) |
| `LR` | 5e-6 | Learning rate |
| `EPOCHS` | 5 | Number of training epochs |
| `TP_SIZE` | 8 | vLLM tensor parallel size |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [verl](https://github.com/volcengine/verl) — Reinforcement learning framework for LLM training
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM serving engine
- [OmniGAIA](https://huggingface.co/datasets/RUC-NLPIR/OmniGAIA) — Omnimodal benchmark dataset
