
<!-- # SliME -->

# UIPro: Unleashing Superior Interaction Capability For GUI Agents (ICCV 2025)

![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
<a href='https://arxiv.org/abs/2406.08487'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/collections/yifanzhang114/slime-665bcb2d0d71762b86fdbd2d'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/yifanzhang114/SMR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green'></a>

<div align="center">
<img src="assets/uipro_github_banner.png" alt="UIPro Project Banner">
</div>


Official Repository for the UIPro research paper from ICCV 2025.
UIPro is a novel generalist GUI agent that represents a breakthrough in GUI automation. It is capable of perceiving and operating graphical user interfaces across multiple platforms, just like a human would. By training on a vast, multi-platform, multi-task dataset and using a unified action space, UIPro achieves state-of-the-art performance on a wide range of GUI interaction and grounding benchmarks.

✨ Key Highlights
<!-- 🌐 Multi-Platform Mastery: UIPro is designed to work seamlessly across web browsers, Android devices, and iPads. -->

📊 Massive-Scale Training: The agent is trained on a foundational dataset of 20.6 million GUI understanding tasks from 2.5 million unique screenshots.

🎮 Unified Action Space: We introduce an innovative framework that harmonizes heterogeneous GUI interactions into a coherent and consistent action space.

🏆 State-of-the-Art Performance: UIPro sets a new standard by achieving superior results across multiple GUI benchmarks, surpassing existing methods.

🔬 Two Model Variants: We provide two model variants, UIPro-SLiME (3B) and UIPro-Qwen2VL (7B), to suit different computational needs.

🏗️ Architecture Overview
UIPro is built upon a proven vision-language model architecture. This architecture allows it to understand instructions, interpret visual information, and plan precise actions.

<div align="center">
<img src="assets/uipro_mainfigure.png" alt="UIPro Methodology Diagram">
<br>
The two-stage training process for UIPro.
</div>

Training Pipeline:

GUI Understanding Pre-training: We first pre-train UIPro on our massive, cleaned dataset of GUI understanding tasks. This instills a strong capability for grounding elements based on descriptions and user intent.

Unified Agent Task Fine-tuning: The pre-trained model is then fine-tuned on a merged dataset of GUI agent tasks using our proposed unified action space. This process enables the agent to predict and execute a sequence of actions to complete complex user tasks.

🎯 Capabilities Showcase
GUI Understanding
Element Grounding: Accurately locates UI elements based on descriptions.

Functionality Recognition: Understands the purpose and function of each interface component.

Intent Mapping: Connects user intentions to the appropriate UI interactions.

GUI Agent Tasks
Task Planning: Breaks down complex user requests into a sequence of actionable steps.

Action Execution: Performs clicks, typing, scrolling, and gestures with high precision.

Cross-Platform Navigation: Operates seamlessly across different device types without requiring platform-specific re-training.

📊 Performance Benchmarks
UIPro demonstrates exceptional performance on industry-standard benchmarks. The tables below show a comparison of UIPro's Step Success Rate (Step SR) and grounding accuracy against other leading methods.

GUI Agent Task Evaluation (AITW Benchmark)

| Benchmark | UIPro-SLiME (3B) | UIPro-Qwen2VL (7B)  |
|-----------|-------------------|---------------------|
| AITW | **68.0%** | **70.4%** |
| AndroidControl | **61.1%** | **85.5%** |
| GUIAct-Web | **68.2%** | **69.1%**  |
| Mind2Web | **28.7%** | **48.4%** |

*Step Success Rate (Step SR) - Higher is better*


## 🚀 Quick Start
Installation
Clone the repository and install the required dependencies.

### Clone the repository
```
git clone https://github.com/ZJULiHongxin/UIPro.git
cd UIPro
```

### Install dependencies
```pip install -r requirements.txt```


## Basic Usage
Here is a simple example of how to use the UIPro model to predict an action.

from uipro import UIPro

### Initialize the model
model = UIPro.from_pretrained("uipro-qwen2vl-7b")


## 📚 Dataset: The Foundation of Excellence
Our training dataset is the largest and most comprehensive GUI understanding collection available. Key statistics include:

20.6M GUI understanding task samples

2.5M unique GUI screenshots

3.3M clean GUI elements

13 different task types

We also implemented a systematic denoising procedure to ensure data quality, removing up to 29% of noise from some data sources.

## 🔬 Technical Deep Dive
Unified Action Space Design
Our unified action space resolves conflicts between different GUI interaction frameworks by defining a single, coherent superset of actions.

Example: Mobile Actions
```
{
  "mobile_actions": [
    "tap", "long_press", "drag", "input_text",
    "navigate_home", "navigate_back", "navigate_recent",
    "press_enter", "swipe", "wait", "status_complete"
  ]
}
```

This framework standardizes action arguments, like the unified swipe action:

### Example: Unified swipe action
```
{
  "action": "swipe",
  "start": [x, y],          // Starting coordinates
  "direction": "up",        // Movement direction
  "distance": 200           // Swipe distance in pixels
}
```

# 📝 Citation
If you use UIPro in your research, please consider citing our paper:
```
@inproceedings{li2025uipro,
  title={UIPro: Unleashing Superior Interaction Capability For GUI Agents},
  author={Li, Hongxin and Su, Jingran and Chen, Jingfan and Ju, Zheng and Chen, Yuntao and Li, Qing and Zhang, Zhaoxiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
# 👥 Team & Acknowledgments
This work was supported in part by the National Key R&D Program of China and the National Natural Science Foundation of China. We also thank the open-source community for providing foundational datasets and tools that made this research possible.

<div align="center">
<br>
<br>
⭐ Star this repository if you find UIPro helpful! ⭐
<br>
<br>
</div>