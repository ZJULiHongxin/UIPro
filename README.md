<div align="center">

# 🚀 UIPro: Unleashing Superior Interaction Capability For GUI Agents

### 🎯 ICCV 2025 • Next-Generation AI GUI Automation

<p align="center">
  <img src="https://img.shields.io/badge/🔬_Research-ICCV_2025-FF6B6B?style=for-the-badge&labelColor=1A1A2E&color=FF6B6B" alt="Research Badge"/>
  <img src="https://img.shields.io/badge/🤖_AI-Multi--Modal-00D4FF?style=for-the-badge&labelColor=1A1A2E&color=00D4FF" alt="AI Badge"/>
  <!-- <img src="https://img.shields.io/badge/⚡_Performance-SOTA-FFD93D?style=for-the-badge&labelColor=1A1A2E&color=FFD93D" alt="Performance Badge"/> -->
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/ICCV2025/html/Li_UIPro_Unleashing_Superior_Interaction_Capability_For_GUI_Agents_ICCV_2025_paper.html">
    <img src="https://img.shields.io/badge/📄_Paper-arXiv-B892FF?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1A1A2E" alt="Paper"/>
  </a>
  <!-- <a href="https://huggingface.co/collections/yifanzhang114/slime-665bcb2d0d71762b86fdbd2d">
    <img src="https://img.shields.io/badge/🤗_Models-Hugging_Face-FFB86C?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=1A1A2E" alt="Models"/>
  </a> -->
  <!-- <a href="https://huggingface.co/datasets/yifanzhang114/SMR">
    <img src="https://img.shields.io/badge/📊_Dataset-Hugging_Face-6BCF7F?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=1A1A2E" alt="Dataset"/>
  </a> -->
</p>

<img src="assets/uipro_github_banner.png" alt="UIPro Project Banner" style="border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin: 20px 0;"/>

</div>

## 📢 News

- **[2025-06-28]** 🎉 UIPro has been accepted to **ICCV 2025**!
- **[2025-11-23]** Uploaded [UIPro Grounding models](https://huggingface.co/HongxinLi/UIPro-7B_Stage1).
- **[2025-11-23]** Uploaded data processing scripts, and systematic denoising procedures for AITW, AITZ, MobileViews, WAE, WebUI, MultiUI, AndroidControl, GUIOdyssey, AMEX, GUIAct
- **[2025-11-27]** Uploaded data processing scripts, and systematic denoising procedures for SeeClick-Web, RefExp
- **[2025-12-10]** Uploaded data processing scripts, and systematic denoising procedures for MOTIF, RefExp, and GUIEnv
- **[2025-12-15]** Uploaded [UIPro Web Agent model](https://huggingface.co/HongxinLi/UIPro-7B_Stage2_Web) and [UIPro Mobile Agent model](https://huggingface.co/HongxinLi/UIPro-7B_Stage2_Mobile).
- **[2026-02-03]** Uploaded evaluation scripts.
- **[2026-05-21]** Released the processed [AndroidControl dataset](https://huggingface.co/datasets/HongxinLi/UIPro-AndroidControl-Data-v1) on HuggingFace. The script now generates **TextLoc / OCR / WidgetList** samples in addition to ActionPlanning / IntentGnd, with WAE-style invalid-element recording, leaf-only + status-bar ambiguity filters, per-sample point↔bbox format toggle (`PROB_BOX=0.3`), and a per-episode pickle cache that skips TensorFlow on re-runs.
- **[2026-09-28]** Built and released the processed [RICO-SCA](https://huggingface.co/datasets/HongxinLi/UIPro-RICOSCA-Data-v1) and [Widget Captioning](https://huggingface.co/datasets/HongxinLi/UIPro-WidgetCaptioning-Data-v1) datasets, and linked all published training and evaluation dataset URLs in the README.

- **[TODO]** Upload whole datasets

---

<div align="center">

## 🌟 **Revolutionary GUI Agent Technology**

*UIPro represents a paradigm shift in GUI automation, achieving human-level interaction capabilities across multiple platforms through advanced AI.*

</div>

### 🎨 **What Makes UIPro Special**

<div align="center">

## 🏗️ **Architecture & Training Pipeline**

*A two-stage revolutionary approach to GUI agent development*

<img src="assets/uipro_mainfigure.png" alt="UIPro Methodology Diagram" style="width: 100%; border-radius: 10px; margin: 20px 0;"/>

### 🔄 **Two-Stage Training Process**

</div>

The training process involves a sophisticated pipeline designed to enhance both understanding and execution capabilities.

---

<div align="center">

## 🎯 **Core Capabilities**

</div>

### 🧠 GUI Understanding Capabilities

| Capability | Description | Performance |
| :--- | :--- | :--- |
| **🎯 Element Grounding** | Accurately locates UI elements based on descriptions | ⭐⭐⭐⭐⭐ |
| **🔍 Functionality Recognition** | Understands purpose and function of interface components | ⭐⭐⭐⭐⭐ |
| **🧭 Intent Mapping** | Connects user intentions to appropriate UI interactions | ⭐⭐⭐⭐⭐ |

### 🤖 GUI Agent Task Execution

| Capability | Description | Performance |
| :--- | :--- | :--- |
| **📋 Task Planning** | Breaks down complex requests into actionable steps | ⭐⭐⭐⭐⭐ |
| **⚡ Action Execution** | Performs clicks, typing, scrolling with high precision | ⭐⭐⭐⭐⭐ |
| **🌐 Cross-Platform Navigation** | Seamless operation across different device types | ⭐⭐⭐⭐⭐ |

---

<div align="center">

## 📊 **Performance Benchmarks**

*Industry-leading results across all major GUI benchmarks*

### 🏆 **GUI Agent Task Evaluation**

| 🎯 Benchmark | 🤖 UIPro-SLiME (3B) | 🚀 UIPro-Qwen2VL (7B) | 📊 Metric |
| :---: | :---: | :---: | :---: |
| **AITW** | <span style="color: #00D4FF; font-weight: bold;">68.0%</span> | <span style="color: #FF6B6B; font-weight: bold;">70.4%</span> | Step SR |
| **AndroidControl** | <span style="color: #00D4FF; font-weight: bold;">61.1%</span> | <span style="color: #FF6B6B; font-weight: bold;">85.5%</span> | Step SR |
| **GUIAct-Web** | <span style="color: #00D4FF; font-weight: bold;">68.2%</span> | <span style="color: #FF6B6B; font-weight: bold;">69.1%</span> | Step SR |
| **Mind2Web** | <span style="color: #00D4FF; font-weight: bold;">28.7%</span> | <span style="color: #FF6B6B; font-weight: bold;">48.4%</span> | Step SR |

<small><i>Step Success Rate (Step SR) - Higher is better</i></small>

</div>

---



## 🚀 **Quick Start Guide**

*Get up and running with UIPro in minutes*

</div>

### 📦 **Installation**

<details>
<summary><b>🔧 Setup Instructions</b></summary>
<br>

#### 1️⃣ **Clone Repository**

```bash
git clone https://github.com/ZJULiHongxin/UIPro.git
cd UIPro
```

#### 2️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

</details>

<!-- ### 💡 **Basic Usage**

<details>
<summary><b>🎮 Quick Example</b></summary>
<br>

```python
from uipro import UIPro

# Initialize the model
model = UIPro.from_pretrained("uipro-qwen2vl-7b")

# Your GUI automation code here
```

</details> -->

---

<div align="center">

## 📚 **Dataset: The Foundation of Excellence**

*The world's largest and most comprehensive GUI understanding collection*

</div>

<div align="center">

### 📊 **Dataset Statistics**

| Metric | Value | Description |
| :---: | :---: | :---: |
| 📝**Task Samples** | **20.6M** | GUI understanding tasks |
| 🖼️**Screenshots** | **2.5M** | Unique GUI screenshots |
| 🎯**Elements** | **3.3M** | Clean GUI elements |
| 🔢**Task Types** | **13** | Different task categories |

</div>

### 🏗️ **Data Compilation Pipeline**

We provide comprehensive scripts to process various GUI datasets. Please follow the instructions below for each dataset.

> **Note**: We also implemented a systematic denoising procedure to ensure data quality, removing up to 29% of noise from some data sources.

#### 🤗 Published Datasets on HuggingFace

The processed datasets below are released publicly under [`HongxinLi`](https://huggingface.co/HongxinLi). Each repo bundles the full samples `.json`, a random `_sample.json` preview, an `_info.json` report, and the referenced images (`_images.zip`).

| Dataset | Stage | HuggingFace |
| :--- | :---: | :--- |
| MobileViews | Understanding | [UIPro-MobileViews-SFT-v1](https://huggingface.co/datasets/HongxinLi/UIPro-MobileViews-SFT-v1) |
| WebUI | Understanding | [UIPro-WebUI-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-WebUI-Data-v1) |
| MultiUI | Understanding | [UIPro-MultiUI-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-MultiUI-Data-v1) |
| SeeClick-Web | Understanding | [UIPro-SeeClickWeb-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-SeeClickWeb-Data-v1) |
| GUIEnv (GUICourse) | Understanding | [UIPro-GUIEnv-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-GUIEnv-Data-v1) |
| RefExp | Understanding | [UIPro-RefExp-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-RefExp-Data-v1) |
| MOTIF | Understanding | [UIPro-MOTIF-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-MOTIF-Data-v1) |
| RICO-SCA | Understanding | [UIPro-RICOSCA-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-RICOSCA-Data-v1) |
| Widget Captioning | Understanding | [UIPro-WidgetCaptioning-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-WidgetCaptioning-Data-v1) |
| AndroidControl | Agent (CFT) | [UIPro-AndroidControl-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-AndroidControl-Data-v1) |

> Each dataset's own section below repeats its link in the **Publishing to HuggingFace** step, alongside the exact command used to build it.

#### GUI Understanding SFT Data Processing

<details>
<summary><b>📱 MobileViews</b></summary>
<br>

> **Prerequisite**: The processing script uses `pytesseract` for overlap-filtering OCR, which requires the system-level `tesseract` binary (the `pytesseract` PyPI package alone is not enough). Install it before running:
> ```bash
> # Debian/Ubuntu
> sudo apt-get update && sudo apt-get install -y tesseract-ocr
> # Or, into the active conda env without sudo:
> conda install -c conda-forge tesseract
> ```
> Verify with `tesseract --version`. Without it the script crashes with `TesseractNotFoundError`.

1. Download the MobileViews raw data from [HuggingFace](https://huggingface.co/datasets/mllmTeam/MobileViews) via:
   ```bash
   hf download mllmTeam/MobileViews --repo-type dataset --local-dir ./MobileViews

   # Move the zipped image files and csv files under the created MobileViews folder
   cd MobileViews
   mv ./MobileViews_Screenshots_ViewHierarchies/Index_CSV/* .
   mv MobileViews_Screenshots_ViewHierarchies/Zip_Files/* ./
   ```

2. Unzip and organize the data as follows:
   ```
   root/
   ├── MobileViews/
   │   ├── MobileViews_0-150000/
   │   ├── MobileViews_index_0-150000.csv
   │   ├── MobileViews_150001-291197/
   │   ├── MobileViews_index_150001-291197.csv
   │   └── ...
   ```
   
3. Modify the top-of-file configuration in `utils/data_utils/make_mobileviews_data/extract_and_generate_mobilebiews_data.py`:
   - **Paths**: `MOBILEVIEWS_DIR`, `ROOT`.
   - **Coordinates**: `SCALE` (coordinate scale), `PROB_BOX` (proportion of box-prediction samples).
   - **Task switches**: `TEXTLOC`, `OCR`, `INTENTGND`, `WIDGETLIST` (enable/disable per-task sample generation).
   - **Parallelism**: `NUM_WORKERS` (default = `cpu_count − 1`). Set to `1` to force the sequential code path — useful when debugging or when the tesseract subprocess load thrashes the machine.

4. Run the processing script. With multiprocessing the wall-clock time drops sharply versus the previous single-process build:
   ```bash
   python utils/data_utils/make_mobileviews_data/extract_and_generate_mobilebiews_data.py
   ```
   Processed training samples land in `ROOT/mobileviews_processed/` along with:
   - `mobileviews_<flags>_scale<S>_<N>k.json` — the full dataset.
   - `…_sample.json` — a small random preview for quick inspection.
   - `…_info.json` — a structured report with `images` / `elements` / `samples.by_task` / `per_part` / `top_packages` / `errors_summary` sections plus a plain-text `report` field. Watch for the loud `⚠ N worker exceptions caught` block at the end of stdout — if it shows up with `[HIGH]`, check `errors_summary` in the info JSON before trusting the counts.

5. **Publishing to HuggingFace**: bundle the produced JSON + referenced images into a dataset repo with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-MobileViews-SFT-v1` → [https://huggingface.co/datasets/HongxinLi/UIPro-MobileViews-SFT-v1](https://huggingface.co/datasets/HongxinLi/UIPro-MobileViews-SFT-v1)).

</details>

<details>
<summary><b>📱 WAE</b></summary>
<br>

1. Download the WAE raw data from [WAE DropBox](https://www.dropbox.com/scl/fo/im6hs48z43h86i80xr517/APE_cMh8qVX_l1Jf_kqArRA/ui30k?dl=0&rlkey=etueluh3jw5adnnntcgrqnqqq&subfolder_nav_tracking=1).

2. Merge, unzip and organize the data as follows:
   ```
   root/
   ├── WAE/
   │   ├── COM.HSBFREE_25-output
   │   ├── Com.sktelecom.minit_52-output
   │   ├── Draziw.Button.Mines_71-output
   │   ├── HBVerbrauchszaehler.lite_119-output
   │   └── ...
   ```

3. Modify `WAE_DIR`, `ROOT_DIR`, `SCALE` (coordinate scale), and `PROB_BOX` (proportion of the box-prediction samples) in `utils/data_utils/make_WAE_data/make_WAE_data.py`.

4. Run the processing script (this may take ~24 hours due to the large number of screenshots):
   ```
   python utils/data_utils/make_WAE_data/make_WAE_data.py
   ```
   Processed training samples will be saved in `ROOT/WAE_processed`.

</details>

<details>
<summary><b>📱 WebUI</b></summary>
<br>

1. Download the WebUI raw data from [biglab/webui-all](https://huggingface.co/datasets/biglab/webui-all).

2. Merge, unzip and organize the data as follows:
   ```
   root/
   ├── WebUI/
   │   ├── dh2 (GUI metadata resulting from unzipping)
   │   ├── WebUI_screenshots (A folder used to save processed GUI screenshots)
   ```

3. Modify the top-of-file configuration in `utils/data_utils/make_webui_data/make_webui_data.py`:
   - **Paths**: `WEBUI_DIR`, `WEBUI_PROCESSED_IMG_DIR`, `ROOT`.
   - **Coordinates**: `SCALE`, `PROB_BOX`, `LONGEST` (longest-edge resize, `-1` disables).
   - **Task switches**: `ICONGND` / `ICONREF` / `ELEMGND` / `ELEMREF` / `ELEMCLASS` / `INTENTGND` / `WIDGETLIST` — toggle the tasks you want generated. `ELEMCLASS=True` also triggers a class-balancing pass that down/upsamples to the 75th percentile.
   - **Parallelism**: `NUM_WORKERS` (default = `cpu_count − 1`) and `CHECKPOINT_EVERY` (how often `invalid_elem_record.json` is flushed during the run). Set `NUM_WORKERS=1` or `DRAW=True` to force the sequential path.
   - **Debug**: `DEBUG=True` processes only the first 1000 sample-dirs — recommended for an initial smoke test.

4. Run the processing script:
   ```bash
   python utils/data_utils/make_webui_data/make_webui_data.py
   ```
   With multiprocessing enabled the long tail of ~348k sample-dirs completes in hours rather than days. Outputs land in `ROOT/WebUI_processed/`:
   - `WebUI_scale<S>_<N>k.json` — the full dataset (samples have globally-renumbered IDs).
   - `…_sample.json` — random preview (160 samples).
   - `…_info.json` — structured report with `images` / `elements` (cumulative) / `samples.by_task` / `run_stats` (per-run) / `elem_class_stats_before_rebalance` / `errors_summary` / plain-text `report`.

5. **Persistent state across runs**. The script keeps two cumulative files in `ROOT/WebUI_processed/` so the reported ratios stay coherent across multiple invocations:
   - `invalid_elem_record.json` — set of all sample identifiers ever marked invalid (used for the cumulative invalid count).
   - `cumulative_stats.json` — cumulative `iterated_elem_cnt` / `processed_img_cnt` / `valid_img_cnt`.

   **Delete both files together** to reset the lifetime tallies. If you only delete one you'll see a `⚠ BUG` flag in the ratio cell because the numerator and denominator will be on different scopes.

6. **Watching for regressions**. The script prints a grouped error summary at the end of every run:
   ```
   ⚠ 2,143 worker exceptions caught (0.6% of sample-dirs). Top error patterns:
       2,143  UnboundLocalError: cannot access local variable 'loc' …
   ```
   If you see `[HIGH]` (>1% of sample-dirs errored) or the sample count drops sharply from a previous run, check `errors_summary` in `_info.json` for the root cause before training on the output.

7. **Publishing to HuggingFace**: bundle the produced JSON + referenced images into a dataset repo with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-WebUI-Data-v1` → [https://huggingface.co/datasets/HongxinLi/UIPro-WebUI-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-WebUI-Data-v1)).

</details>

<details>
<summary><b>📱 MultiUI</b></summary>
<br>

1. Download the MultiUI raw data from [neulab/MultiUI](https://huggingface.co/datasets/neulab/MultiUI) via `hf download neulab/MultiUI --repo-type=dataset --local-dir ./MultiUI`.

2. Merge, unzip and organize the data as follows:
   ```
   root/
   ├── MultiUI/
   │   ├── v0.6_5M
   │   ├── v0.7_exclude_v0.6
   |   ├── v0.8_exclude_v0.7
   │   ├── stage1_data.json
   │   ├── stage1_data_10k.json
   |   ├── stage2_data_to_be_combined_with_general_data.json
   ```

3. Modify the top-of-file configuration in `utils/data_utils/make_multiui_data/make_multiui_data.py`:
   - **Paths**: `MULTIUI_SAMPLE_FILE`, `IMG_DIR`, `SAVE_ROOT`.
   - **Coordinates**: `SCALE` (coordinate scale).
   - **Task switches**: `ENABLE_UI_CAPTION` / `ENABLE_WEBQA` / `ENABLE_OCR` / `ENABLE_GROUNDING` — toggle which task families get generated.
   - **Parallelism**: `NUM_WORKERS` (default = `cpu_count − 1`). Set to `1` to force the sequential code path when debugging.
   - **Debug**: `DEBUG=True` processes a sampled subset (~1/1250 of rows) for a quick smoke test.

4. Run the processing script (the per-sample image-dimension I/O dominates wall-clock time, so multiprocessing gives a large speed-up):
   ```bash
   python utils/data_utils/make_multiui_data/make_multiui_data.py
   ```
   Two output splits land in `ROOT/MultiUI_processed/`, each with a unified sibling `_info.json` (structured `images` / `elements` / `samples.by_task` / `errors_summary` / report sections) and a `_sample.json` preview:
   - `MultiUI_gnd_ref_qa_scale<S>_<N>k.json` — grounding / QA / OCR samples.
   - `MultiUI_caption_<N>k.json` — UI caption samples (only if `ENABLE_UI_CAPTION=True`).

5. **Quality-control notes**:
   - The grounding path performs a **blank-element check**: bounding-boxes whose pixel region is a pure colour are dropped and counted in `elements.invalid_breakdown.blank_elem`.
   - Failed query extractions (e.g. malformed prompts) land in `invalid_breakdown.noquery_elem`.
   - The cumulative `invalid_elem` IDs are persisted to `stats.json` so subsequent runs skip them automatically.
   - Watch stdout for the `⚠ N worker exceptions caught` block at the end; `[HIGH]` (>1% of tasks) usually means a config/path problem worth fixing before training.

6. **Publishing to HuggingFace**: bundle the produced JSON + referenced images into a dataset repo with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-MultiUI-Data-v1` → [https://huggingface.co/datasets/HongxinLi/UIPro-MultiUI-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-MultiUI-Data-v1)).

</details>

<details>
<summary><b>📱 SeeClick-Web</b></summary>
<br>

1. Download the SeeClick-Web raw data from [SeeClick-Web Annotation File](https://box.nju.edu.cn/f/3b0f6ccb8bed476c8e39/) and [SeeClick-Web Images](https://box.nju.edu.cn/f/6a804cf190dd490a808f/).

2. Unzip and organize the data as follows:
   ```
   root/
   ├── SeeClick-Web/
   │   ├── 0a5c8a5b7d73de574f2a21f27dbc9a53.png
   │   ├── 0a6dcd3f9e1907af232e2c038a866f74.png
   | ...
   ```

3. Modify the top-of-file configuration in `utils/data_utils/make_seeclickweb_data/make_seeclickweb_data.py`:
   - **Paths**: `IMG_DIR`, `ANNO_FILE`, `SAVE_ROOT`.
   - **Coordinates**: `SCALE` (coordinate scale).
   - **Task switches**: `ELEMGND` (element grounding: text → location) and `ELEMREF` (element referring: location → text). When both are `True`, each valid element yields two samples (one per task). `USE_ACTION_PROMPT=True` overrides both and emits a single action-planning sample instead.
   - **Validation**: `SKIP_CHECKING=True` bypasses the pure-colour / duplicate / size checks for a faster, lower-quality run.
   - **Parallelism**: `NUM_WORKERS` (default = `cpu_count − 1`) and `CHECKPOINT_EVERY` (how often `invalid_elem_record.json` is flushed). Set `NUM_WORKERS=1` to force sequential.

4. Run the processing script (multiprocessing brings the wall-clock from ~24 h down to a few hours on a typical workstation):
   ```bash
   python utils/data_utils/make_seeclickweb_data/make_seeclickweb_data.py
   ```
   Outputs land in `ROOT/SeeClick-Web_processed/` and the filename encodes the enabled tasks so different flag combinations don't overwrite each other:
   - `SeeClick-Web_ElemGnd_ElemRef_<N>k.json` (both tasks)
   - `SeeClick-Web_ElemGnd_<N>k.json` / `SeeClick-Web_ElemRef_<N>k.json` (one task)
   - `SeeClick-Web_actformat_<N>k.json` (action-planning shortcut)

   Each output gets a sibling `_sample.json` preview and a `_info.json` with structured `images` / `elements` / `samples.by_task` / `errors_summary` sections plus a plain-text `report`.

5. **Persistent state across runs**: the script keeps `invalid_elem_record.json` in `SAVE_ROOT` — a cumulative skip-list of sample identifiers that have already been rejected. Re-runs benefit automatically (O(1) lookup at start, broadcast to every worker via a Pool initializer). Delete the file to reset.

6. **Publishing to HuggingFace**: bundle the produced JSON + referenced images into a dataset repo with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-SeeClickWeb-Data-v1` → [https://huggingface.co/datasets/HongxinLi/UIPro-SeeClickWeb-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-SeeClickWeb-Data-v1)).

</details>

<details>
<summary><b>📱 GUIEnv (GUICourse subtask)</b></summary>
<br>

> **Prerequisite**: GUIEnv runs `pytesseract` against every candidate element to validate OCR annotations. Install the `tesseract` system binary first (see the [MobileViews prerequisite block](#-mobileviews) for instructions) — `pip install pytesseract` alone is not enough.

All three GUICourse subtasks (GUIEnv / GUIChat / GUIAct) are handled by the same script (`utils/data_utils/make_guicourse_data/make_guicourse_data.py`) under one `DatasetConfig`; the `SUBTASK` list controls which subtasks run in a given invocation.

1. Download the GUIEnv raw data from [yiye2023/GUIEnv](https://huggingface.co/datasets/yiye2023/GUIEnv).

2. Unzip and organise the data as follows:
   ```
   root/
   │   └── GUICourse/
   │            └── GUIEnv/
   │                    ├── imgs
   │                    ├── ocr_grounding_train_stage2_images.parquet
   │                    ├── ocr_grounding_train_stage2_data.json
   │                    ├── ocr_grounding_train_stage1_images.parquet
   │                    ├── ocr_grounding_train_stage1_data.json
   │                    ├── ocr_grounding_test_images.parquet
   │                    └── ocr_grounding_test_data.json
   ```

3. Edit the `DatasetConfig` class at the top of `utils/data_utils/make_guicourse_data/make_guicourse_data.py`:
   - **Subtask selection**: `SUBTASK = ['GUIEnv']` (add `'GUIChat'` / `'GUIAct'` to run more in one shot).
   - **Paths**: `DATA_ROOT`, `SAVE_DIR`.
   - **Coordinates**: `SCALE`, `BOUNDING_BOX_PROBABILITY`.
   - **Split**: `CURRENT_SPLIT` (`'train'` / `'test'`).
   - **Task switches**: `ENABLE_TEXTLOC`, `ENABLE_OCR` — toggle which of the two per-element variants get emitted.
   - **Validation**: `SKIP_VALIDATION=True` bypasses the size / box / pure-colour / OCR-similarity checks for a faster, lower-quality run.
   - **Parallelism**: `NUM_WORKERS` (default = `cpu_count − 1`). Parquet read + base64 image decoding stay on the main process (cheap, one-time per shard); the per-image-group validation and sample-construction work fans out across the pool. Set `NUM_WORKERS=1` to force the sequential code path.
   - **Debug**: `DEBUG_MODE=True` only matters for GUIAct (sub-samples 1-in-100 trajectories).

4. Run the script:
   ```bash
   python utils/data_utils/make_guicourse_data/make_guicourse_data.py
   ```
   Outputs land in `SAVE_DIR`:
   - `guienv-<split>-s<SCALE>-<N>.json` — full dataset (sample IDs renumbered globally per-prefix after parallel aggregation).
   - `…_sample.json` — 256-sample preview.
   - `…_info.json` — structured report with `images` / `elements` (with `invalid_breakdown`) / `samples.by_task` (`TextLoc` + `OCR`) / `errors_summary` / plain-text `report`.

5. **Persistent state across runs**: the script keeps `invalid_elem_record.json` in `SAVE_DIR` — a cumulative skip-list of sample identifiers that have been rejected (too-small box, blank region, OCR mismatch, …). Broadcast to every worker via a Pool initialiser for O(1) lookups; delete the file to reset.

6. **Watching for regressions**: a `⚠ N worker exceptions caught` block is printed at the end of every run; the `[HIGH]` flag (>1% of items) usually means a path / permission / encoding issue worth fixing before training.

7. **Publishing to HuggingFace**: bundle the produced JSON + referenced images into a dataset repo with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-GUIEnv-Data-v1` → [https://huggingface.co/datasets/HongxinLi/UIPro-GUIEnv-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-GUIEnv-Data-v1)).

</details>

<details>
<summary><b>📱 GUIChat (GUICourse subtask)</b></summary>
<br>

Conversational GUI samples with bounding-box annotations. Generated by the same `make_guicourse_data.py` script with `SUBTASK` including `'GUIChat'`.

1. Download the GUIChat parquet shards (and matching `data.json` files) from the [GUICourse](https://huggingface.co/datasets/yiye2023/GUICourse) collection and place them at `DATA_ROOT/GUIChat/*.parquet`.

2. Add `'GUIChat'` to `DatasetConfig.SUBTASK`. The chat path doesn't have its own task switches — `SCALE` (coordinate scale) and `NUM_WORKERS` are the only knobs that affect output.

3. Run the script:
   ```bash
   python utils/data_utils/make_guicourse_data/make_guicourse_data.py
   ```
   Per-image cv2.imread + bounding-box normalisation is the slow path; parallelisation gives a meaningful speed-up here. Outputs land in `SAVE_DIR`:
   - `guichat-<split>-s<SCALE>-<N>.json` (full dataset) + `_sample.json` (256-row preview) + `_info.json` (unified report with `samples.by_task.WebQA` + `errors_summary`).

</details>

<details>
<summary><b>📱 RefExp</b></summary>
<br>

> **Prerequisite**: RefExp uses spaCy's medium English model (`en_core_web_md`) to classify each instruction as "element" (no verb → element-grounding task) or "intent" (has an action verb → intent-grounding task). Install it once:
> ```bash
> python -m spacy download en_core_web_md
> ```
> A pre-flight check in the script fails fast with this exact command if the model is missing — without it, every Pool worker would crash with `OSError: [E050] Can't find model 'en_core_web_md'`.

1. Download the RICO image data from [SeeClick RICO Data](https://box.nju.edu.cn/f/7ae5e9bd4bf840d4add3/).

2. Unzip and organize the data as follows:
   ```
   root/
   │   └── rico/
   │       ├── 72197.jpg
   │       ├── ...
   │       ├── 71949.json
   │       └── ...

   ```

3. Modify the top-of-file configuration in `utils/data_utils/make_refexp_data/make_refexp_data.py`:
   - **Paths**: `IMG_DIR`, `SAVE_ROOT`.
   - **Coordinates**: `SCALE`, `OUTPUT_TAG` (optional output-format hint appended to the prompt).
   - **Task format**: `USE_ACTION_PROMPT=True` swaps intent grounding for action-planning samples (Aguvis format).
   - **Validation**: `SKIP_CHECKING=True` skips the size / box / pure-colour checks.
   - **Parallelism**: `NUM_WORKERS` (default = `cpu_count − 1`). Each worker loads one `VerbExtactor` (spaCy) at startup — that's a ~1-2 s per-worker cost paid once. Set `NUM_WORKERS=1` to force sequential while debugging.
   - **Debug**: `DEBUG=True` processes only the first 200 samples.

4. Run the processing script:
   ```bash
   python utils/data_utils/make_refexp_data/make_refexp_data.py
   ```
   Outputs land in `ROOT/RefExp_processed/`:
   - `RefExp_s<S>_<N>k.json` — the full dataset (sample IDs are renumbered globally per-prefix after parallel aggregation, so they're unique and contiguous).
   - `…_sample.json` — random preview for quick inspection.
   - `…_info.json` — structured report with `images` / `elements` / `samples.by_task` (`ElemGnd` + `IntentGnd`) / `errors_summary` / plain-text `report`.

5. **Persistent state across runs** (in `SAVE_ROOT`):
   - `invalid_elem_record.json` — cumulative skip-list of rejected sample identifiers.
   - `instruc_type_record.json` — cache of instruction → first-verb mappings. Seeded into every worker via the Pool initializer so re-runs skip the spaCy parse for known instructions.
   
   Delete either file to reset that cache.

6. **Publishing to HuggingFace**: bundle the produced JSON + referenced images into a dataset repo with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-RefExp-Data-v1` → [https://huggingface.co/datasets/HongxinLi/UIPro-RefExp-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-RefExp-Data-v1)).

</details>

<details>
<summary><b>📱 MOTIF</b></summary>
<br>

1. Download the MOTIF image data from [HongxinLi/MOTIF](https://huggingface.co/datasets/HongxinLi/MOTIF).

2. Unzip and organize the data as follows:
   ```
   root/
   │   └── motif/

   ```

3. Modify the top-of-file configuration in `utils/data_utils/make_motif_data/make_motif_data.py`:
   - **Paths**: `IMG_DIR`, `SAVE_ROOT_DIR`.
   - **Coordinates**: `SCALE`, `OUTPUT_TAG` (optional output-format hint).
   - **Task format**: `USE_ACTION_PROMPT=True` formats samples as action-planning (Aguvis) instead of intent grounding.
   - **Validation**: `SKIP_CHECKING=True` skips the size / box / pure-colour / swipe-filter checks.
   - **Parallelism**: `NUM_WORKERS` (default = `cpu_count − 1`). Set `NUM_WORKERS=1` to force the sequential code path while debugging.
   - **Debug**: `DEBUG=True` processes only the first 200 rows.

4. Run the processing script (file-I/O bound per row, so multiprocessing helps a lot on networked storage):
   ```bash
   python utils/data_utils/make_motif_data/make_motif_data.py
   ```
   Outputs land in `ROOT/MOTIF_processed/`:
   - `MOTIF_<N>k<flags>.json` — full dataset (`<flags>` is `_debug` / `_actformat` when those are set; sample IDs are renumbered globally after parallel aggregation).
   - `…_sample.json` — random preview.
   - `…_info.json` — structured report with `images` / `elements` / `samples.by_task` (`IntentGnd`) / `invalid_breakdown` (`too_small_element`, `blank_elem`, `invalid_elem_content`, …) / `errors_summary` / plain-text `report`.

5. **Persistent state across runs**: the script keeps `invalid_elem_record.json` in `SAVE_ROOT_DIR` — a cumulative skip-list of sample identifiers that have already been rejected. The Pool initializer broadcasts it as a frozenset to every worker for O(1) lookups. Delete the file to reset.

6. **Publishing to HuggingFace**: bundle the produced JSON + referenced images into a dataset repo with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-MOTIF-Data-v1` → [https://huggingface.co/datasets/HongxinLi/UIPro-MOTIF-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-MOTIF-Data-v1)).

</details>

<details>
<summary><b>📱 RICO-SCA</b></summary>
<br>

> **Prerequisite**: This script uses `python-magic` to read image dimensions from file headers, which requires the system-level `libmagic` library (the `python-magic` PyPI package alone is not enough). Install it before running:
> ```bash
> # Debian/Ubuntu
> sudo apt-get update && sudo apt-get install -y libmagic1
> # Or, into the active conda env without sudo:
> conda install -c conda-forge libmagic
> ```
> Without it the script crashes at import with `ImportError: failed to find libmagic`.

RICO-SCA provides single-instruction referring/grounding annotations over the RICO screenshots (the same image set used by RefExp and MOTIF), which we convert into intent-grounding samples.

> **Note on the published release**: [HongxinLi/UIPro-RICOSCA-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-RICOSCA-Data-v1) is built from the **train split** of the public RICO-SCA annotations (173k caption→grounding samples over 18k screens). The paper's Table 1 count (295k) additionally counts held-out splits and multi-phrasing expansion; the published train release is the reproducible core used for the understanding stage.

1. Download the RICO image data from [SeeClick RICO Data](https://box.nju.edu.cn/f/7ae5e9bd4bf840d4add3/) and the RICO-SCA annotation file (`ricosca.json`) from [SeeClick](https://github.com/njucckevin/SeeClick).

2. Unzip and organize the data as follows:
   ```
   root/
   ├── rico/
   │   ├── 72197.jpg
   │   └── ...
   └── ricosca.json
   ```

3. Modify the top-of-file configuration in `utils/data_utils/make_ricosca_data/make_ricosca_data.py`:
   - **Paths**: `IMG_DIR`, `ANNO_FILE`, `SAVE_ROOT`.
   - **Coordinates**: `SCALE` (coordinate scale).
   - **Task format**: `USE_ACTION_PROMPT=True` emits action-planning (Aguvis) samples instead of intent grounding.
   - **Validation**: `SKIP_CHECKING=True` (default) bypasses the swipe-filter / size / box / pure-colour checks for a faster run.
   - **Debug**: `DEBUG=True` processes only the first 200 rows.

4. Run the processing script:
   ```bash
   python utils/data_utils/make_ricosca_data/make_ricosca_data.py
   ```
   Outputs land in `ROOT/RICOSCA_processed/`:
   - `RICOSCA_<N>k<flags>.json` — full dataset (`<flags>` is `_debug` / `_actformat` when set).
   - `…_sample.json` — random preview.
   - `…_info.json` — structured report (`images` / `elements` / `samples.by_task` (`IntentGnd`) / `invalid_breakdown`) plus a plain-text `report`.

5. **Persistent state across runs**: the script keeps `invalid_elem_record.json` in `SAVE_ROOT` — a cumulative skip-list of rejected sample identifiers. Delete it to reset.

6. **Publishing to HuggingFace**: bundle the produced JSON + referenced images with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-RICOSCA-Data-v1`). A pre-processed release (173k intent-grounding samples, train split) is available at [HongxinLi/UIPro-RICOSCA-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-RICOSCA-Data-v1).

</details>

<details>
<summary><b>📱 Widget Captioning</b></summary>
<br>

> **Prerequisite**: Like RICO-SCA, this script uses `python-magic`, which needs the system-level `libmagic` library (see the [RICO-SCA prerequisite block](#-rico-sca) for install instructions). It **additionally** requires spaCy's medium English model, used to classify each caption's leading verb:
> ```bash
> python -m spacy download en_core_web_md
> ```
> Without it the script crashes with `OSError: [E050] Can't find model 'en_core_web_md'`.

Widget Captioning pairs each RICO widget with a natural-language caption, which we turn into element-grounding, intent-grounding, and (optionally) element-referring samples.

> **Note on the published release**: [HongxinLi/UIPro-WidgetCaptioning-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-WidgetCaptioning-Data-v1) is built from the **train split** (109k samples: ~63k intent-grounding + ~46k element-grounding over 15k screens); the paper's Table 1 count (163k) additionally counts held-out splits. The train/val/test splits are kept separate so the eval widgets stay held out.

1. Download the RICO image data from [SeeClick RICO Data](https://box.nju.edu.cn/f/7ae5e9bd4bf840d4add3/) and the Widget Captioning annotation file (`widget_captioning.json`) from the [Widget Captioning dataset](https://github.com/google-research/google-research/tree/master/widget_caption).

2. Unzip and organize the data as follows:
   ```
   root/
   ├── rico/
   │   ├── 72197.jpg
   │   └── ...
   └── widget_captioning.json
   ```

3. Modify the top-of-file configuration in `utils/data_utils/make_widget_captioning_data/make_widcap_data.py`:
   - **Paths**: `IMG_DIR`, `ANNO_FILE`, `SAVE_ROOT`.
   - **Coordinates**: `SCALE` (coordinate scale).
   - **Task switches**: `ELEMREF` toggles element-referring (location → caption) sample generation; element grounding and intent grounding are always produced. `USE_ACTION_PROMPT=True` emits action-planning (Aguvis) samples instead.
   - **Validation**: `SKIP_CHECKING=True` (default) bypasses the size / box / pure-colour checks.
   - **Debug**: `DEBUG=True` processes only the first 200 rows.

4. Run the processing script:
   ```bash
   python utils/data_utils/make_widget_captioning_data/make_widcap_data.py
   ```
   Outputs land in `ROOT/WidgetCaptioning_processed/`:
   - `WidgetCaptioning_<N>k<flags>.json` — full dataset.
   - `…_sample.json` — random preview.
   - `…_info.json` — structured report with per-task counters (`intentgnd_cnt` / `elemgnd_cnt` / `elemref_cnt`) plus a plain-text `report`.

5. **Publishing to HuggingFace**: bundle the produced JSON + referenced images with `utils/data_utils/push_data_to_hf.py` (e.g. `--repo-id HongxinLi/UIPro-WidgetCaptioning-Data-v1`). A pre-processed release (109k grounding samples, train split) is available at [HongxinLi/UIPro-WidgetCaptioning-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-WidgetCaptioning-Data-v1).

</details>

<details>
<summary><b>📱 OmniAct</b></summary>
<br>

1. Download the OmniAct raw data from [Writer/omniact](https://huggingface.co/datasets/Writer/omniact).

2. Unzip and organize the data as follows:
   ```
   root/
   ├── OmniAct/
   │   └── data/
   │       ├── tasks/
   │       ├── metadata/
   │       └── data/
   ```

3. Modify `ROOT_DIR`, `SAVE_ROOT_DIR`, `SPLIT`, and `SCALE` (coordinate scale) in `utils/data_utils/make_omniact_data/make_omniact_data.py`.

4. Run the processing script:
   ```
   python utils/data_utils/make_omniact_data/make_omniact_data.py
   ```
   Processed training samples will be saved in `ROOT/OmniAct_processed`.

</details>

#### Agentic SFT Data Processing

<details>
<summary><b>🤖 Android in the Wild (AiTW)</b></summary>
<br>

1. Download AiTW screenshots from [here](https://box.nju.edu.cn/f/96ba5115bae24eaaa44e/) and annotations from [here](https://box.nju.edu.cn/f/1245c74fc09b4565a235/).
2. Organize the data:
   ```
   root/
   ├── AITW/
   │   ├── aitw_data_test.json
   │   ├── aitw_data_train.json
   │   ├── aitw_data_val.json
   │   └── aitw_images/
   │       ├── general/
   │       ├── googleapps/
   │       ├── install/
   │       ├── single/
   │       └── webshopping/
   ```
3. Modify `ROOT`, `SAVE_DIR`, `SPLIT`, and `POINT_FORMAT` in `utils/data_utils/make_aitw_data/make_aitw_data.py`.
4. Run the script:
   ```bash
   python utils/data_utils/make_aitw_data/make_aitw_data.py
   ```
   Processed samples will be saved in `SAVE_DIR/AITW_processed`.

</details>

<details>
<summary><b>🦓 Android in the Zoo (AitZ)</b></summary>
<br>

1. Download raw data following instructions in the [AitZ Github Repo](https://github.com/IMNearth/CoAT).
2. Organize the data:
   ```
   root/
   ├── AITZ/
   │   ├── train/
   │   │   ├── general/
   │   │   ├── googleapps/
   │   │   ├── install/
   │   │   ├── single/
   │   │   └── webshopping/
   │   └── test/
   │       ├── general/
   │       ├── googleapps/
   │       ├── install/
   │       └── webshopping/
   ```
3. Modify `ROOT`, `SAVE_DIR`, `SCALE`, `SPLIT`, and `USE_ACTION_REFEXP` in `utils/data_utils/make_aitz_data/make_aitz_data.py`.
4. Run the script:
   ```bash
   python utils/data_utils/make_aitz_data/make_aitz_data.py
   ```
   Processed samples will be saved in `SAVE_DIR/AITZ_processed`.

</details>

<details>
<summary><b>🎮 AndroidControl</b></summary>
<br>

1. Download raw data following instructions in the [AndroidControl Github Repo](https://github.com/google-research/google-research/blob/master/android_control/README.md).
2. Organize the data:
   ```
   root/
   ├── AndroidControl/
   │   ├── raw/
   │   │   ├── android_control-00000-of-00020
   │   │   ├── android_control-00001-of-00020
   │   │   ├── ...
   │   │   ├── android_control-00019-of-00020
   │   │   └── splits.json
   ```
3. Edit the top of `utils/data_utils/make_androidcontrol_data/make_androidcontrol_data.py`:
   - **Paths**: `ANDROIDCONTROL_ROOT`, `SAVE_DIR`.
   - **Split / coordinates**: `SPLIT` (`'train'` / `'test'`), `SCALE`, `LONGEST`, `POINT_FORMAT` (`'plain'` / `'qwen'`).
   - **Task switches**:
     - `PLANNING` — high-level and high+low-level action-planning samples (per step).
     - `INTENTGND` — intent-grounding samples for click / long-press steps.
     - `TEXTLOC` / `OCR` / `WIDGETLIST` — element-level samples derived from each step's accessibility tree (added in this update; mirror the WAE / webui task generators).
     - `PROB_BOX = 0.3` — per-sample probability of emitting the answer as a bounding box rather than a center point (applies to TextLoc, OCR, and IntentGnd; the question's `(with bbox)` / `(with point)` suffix tracks the format so prompt and answer never disagree).
     - `USE_ACTION_REFEXP`, `DISCARD_OPENAPP`, `ADD_DUMMY_COMPLETE`, `RETAIN_STATUSBAR_IN_XML`, `DEVICE_TAG` — auxiliary planning knobs.
   - **Parallelism**: `NUM_WORKERS` (default = 8; the per-episode body is dispatched to a `spawn`-method `multiprocessing.Pool` of TF-free workers, so memory cost stays modest). Set `NUM_WORKERS=1` to force sequential / enable `DRAW`.
   - **Debug**: `DEBUG=True` takes the first 30 episodes only.

4. Run the script:
   ```bash
   python utils/data_utils/make_androidcontrol_data/make_androidcontrol_data.py
   ```
   The **first run** walks the gzipped TFRecord stream once and writes a per-episode pickle cache at `ANDROIDCONTROL_ROOT/_episode_cache/{ep_id}.pkl` plus a `_complete.marker` sentinel. **Subsequent runs** skip TensorFlow entirely and load episodes directly from the cache — orders of magnitude faster. Delete `_complete.marker` to force a rebuild.

   Outputs land in `SAVE_DIR` (default `…/AndroidControl/`):
   - `AndroidControl[_test]_s<SCALE>_<N>.json` — full dataset (filename varies with enabled flags).
   - `…_sample.json` — 160-sample preview.
   - `…_info.json` — unified report with `images` / `elements` / `samples.by_task` (`ActionPlanning` + `IntentGnd` + `TextLoc` + `OCR` + `WidgetList`) / `errors_summary` + plain-text `report`.
   - `invalid_elem_record.json` — WAE-style cumulative `{bucket: [sample_id, …]}` record of every rejected node, keyed by `"{short_img_path}|{win_idx}|{box}"`. Buckets use the canonical names from `task_prompt_lib` (`overlapping element`, `invalid box coordinates`, `too small element`, `oversized element`, `invalid element content`, `element not displayed`, plus the AndroidControl-specific `UIs without main app windows`).

5. **Ambiguity & targeting safeguards** (newly tightened): for the three element-level grounding tasks, only **leaf** nodes outside the status-bar window are valid targets, and the per-screen uniqueness check (`<= 1` occurrence in the candidate list) is also restricted to that same leaf set. This prevents (a) container/leaf duplicates (a `View` wrapping a `TextView` with the same content-desc) from falsely flagging unique leaves as ambiguous, and (b) status-bar elements (battery / wifi / clock / notification icons) from dominating the sample distribution. WidgetList is intentionally left alone — it serializes the whole screen as an overview.

6. **Publishing the processed dataset to HuggingFace**: the released artifacts live at [HongxinLi/UIPro-AndroidControl-Data-v1](https://huggingface.co/datasets/HongxinLi/UIPro-AndroidControl-Data-v1). To upload your own run, point `--samples` at the produced JSON and `--images-root` at the parent of the image folders:
   ```bash
   python utils/data_utils/push_data_to_hf.py \
       --samples /path/to/data/AndroidControl/AndroidControl_s1000_<N>.json \
       --images-root /path/to/data \
       --repo-id HongxinLi/UIPro-AndroidControl-Data-v1
   ```

</details>

<details>
<summary><b>🌊 GUIOdyssey</b></summary>
<br>

1. Download raw data from the [GUIOdyssey HF Repo](https://huggingface.co/datasets/hflqf88888/GUIOdyssey).
2. Organize the data:
   ```
   root/
   ├── GUIOdyssey_raw/
   │   ├── screenshots/
   │   │   ├── 2386365564178401_9.png
   │   │   ├── 5022534067657028_12.png
   │   │   ├── 7287738713744873_13.png
   │   │   └── ...
   │   ├── splits/
   │   └── annotations/
   ```
3. Move all images from `data_*` subfolders in `screenshots` directly to `screenshots`.
4. Modify `DATA_ROOT`, `SAVE_ROOT`, and `SPLIT` in `utils/data_utils/make_guiodyssey_data/make_guiodyssey_data.py`.
5. Run the script:
   ```bash
   python utils/data_utils/make_guiodyssey_data/make_guiodyssey_data.py
   ```
   Processed samples will be saved in `SAVE_ROOT/GUIOdyssey_processed`.

</details>

<details>
<summary><b>💳 AMEX</b></summary>
<br>

1. Download raw data from the [AMEX HF Repo](https://huggingface.co/datasets/Yuxiang007/AMEX).
2. Organize the data:
   ```
   root/
   ├── AMEX/
   │   ├── element_anno/
   │   ├── screenshot/
   │   └── metadata/
   ```
3. Modify `DATA_ROOT`, `SAVE_ROOT`, and `SPLIT` in `utils/data_utils/make_amex_data/make_amex_data.py`.
4. Run the script:
   ```bash
   python utils/data_utils/make_amex_data/make_amex_data.py
   ```
   Processed samples will be saved in `SAVE_ROOT/AMEX_processed`.

</details>

<details>
<summary><b>🎭 GUIAct (GUICourse subtask)</b></summary>
<br>

Action-planning samples derived from full task trajectories. Generated by the same `make_guicourse_data.py` script.

1. Download raw data from the [GUIAct HF Repo](https://huggingface.co/datasets/yiye2023/GUIAct):
   ```bash
   hf download yiye2023/GUIAct --repo-type dataset --local-dir path/to/GUICourse/GUIAct
   ```

2. Organise the data:
   ```
   root/
   ├── GUICourse/
   │   ├── GUIAct/
   │   │   ├── smartphone_test_data.json
   │   │   ├── smartphone_test_images.parquet
   │   │   ├── smartphone_train_data.json
   │   │   └── ...
   ```

3. Edit `DatasetConfig` in `utils/data_utils/make_guicourse_data/make_guicourse_data.py`:
   - **Subtask selection**: `SUBTASK = ['GUIAct']`.
   - **Paths**: `DATA_ROOT`, `SAVE_DIR`.
   - **Split / device**: `CURRENT_SPLIT` (`'train'` / `'test'`) and `CURRENT_DEVICE_TYPE` (`'smartphone'` / `'web'`).
   - **Output format**: `USE_ACTION_REFEXP=True` prefixes each generated action with a Qwen-style natural-language reference expression; `ENABLE_INTENT_GND=True` opts into the extra intent-grounding sample per click.
   - **History**: `MAX_PREVIOUS_ACTIONS=3` caps how many prior step instructions are folded into a step's `history` field.
   - **Parallelism**: `NUM_WORKERS` (default = `cpu_count − 1`). **Each trajectory is dispatched to a worker** (steps within a trajectory remain sequential since they share `step_instructions` state), so the speed-up scales with the number of trajectories rather than the number of steps.
   - **Debug**: `DEBUG_MODE=True` keeps every 100th trajectory only — handy for a quick smoke test.

4. Run the script:
   ```bash
   python utils/data_utils/make_guicourse_data/make_guicourse_data.py
   ```
   Outputs land in `SAVE_DIR`:
   - `guiact-<split>-<device>[-wActRef]-s<SCALE>-<N>.json` — full dataset (sample IDs renumbered globally after parallel aggregation).
   - `…_sample.json` — 256-sample preview.
   - `…_info.json` — unified report with `images` / `elements` / `samples.by_task` (`ActionPlanning` + optional `IntentGnd`) / `errors_summary` / plain-text `report`.

5. **Watching for regressions**: same `⚠ N worker exceptions caught` end-of-run block as the other GUICourse subtasks. Trajectory-level failures group by traceback tail in `errors_summary` so a recurrent bug surfaces immediately instead of being buried per-step.

</details>

---

<div align="center">

## 🔬 **Technical Deep Dive**

*Advanced technical details for researchers and developers*

</div>

### 🎮 **Unified Action Space Design**

<details>
<summary><b>📱 Mobile Action Framework</b></summary>
<br>

```json
{
  "mobile_actions": [
    "tap", "long_press", "drag", "input_text",
    "navigate_home", "navigate_back", "navigate_recent",
    "press_enter", "swipe", "wait", "status_complete"
  ]
}
```

</details>

<details>
<summary><b>⚡ Unified Swipe Action</b></summary>
<br>

```json
{
  "action": "swipe",
  "start": [x, y],          // Starting coordinates
  "direction": "up",        // Movement direction  
  "distance": 200           // Swipe distance in pixels
}
```

</details>

---

<div align="center">

## 🧪 **Model Usage**

</div>

**Stage 1: GUI Element Grounding**

Follow the code below to test the [stage 1 model for GUI element grounding](https://huggingface.co/HongxinLi/UIPro-7B_Stage1) evaluation.

```
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "HongxinLi/UIPro-7B_Stage1", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("HongxinLi/UIPro-7B_Stage1")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./web_6f93090a-81f6-489e-bb35-1a2838b18c01.png",
            },

            # For ScreenSpot-v2, MOTIF, RefExp, and VisualWebBench Action Grounding
            {"type": "text", "text": "I want to {goal_info}. Please locate the target element I should interact with. (Output the center coordinates of the target)"},
            # For AutoGUI
            {"type": "text", "text": "Locate the element according to its detailed functionality description. {goal_info} (Output the center coordinates of the target)"},
            # For VisualWebBench Element Grounding
            {"type": "text", "text": "Locate the text "{goal_info}" (Output the center coordinates of the target)"},
        ],
    }
]

```


**Stage 2: Web Agent Embodiment**

Follow the code below to test the [stage 2 model for Web agent](https://huggingface.co/HongxinLi/UIPro-7B_Stage2_Web) task evaluation.

```
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "HongxinLi/UIPro-7B_Stage2_Web", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("HongxinLi/UIPro-7B_Stage2_Web")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./web_6f93090a-81f6-489e-bb35-1a2838b18c01.png",
            },

            {"type": "text", "text": """Given the Web UI screenshot and previous actions, please generate the next move necessary to advance towards task completion. The user's task is: {task}
Action history: {action_history}

Now, first describe the action intent and then directly plan the next action."""},
        ],
    }
]

```


**Stage 2: Mobile Agent Embodiment**

Follow the code below to test the [stage 2 model for Mobile agent](https://huggingface.co/HongxinLi/UIPro-7B_Stage2_Mobile) task evaluation.

```
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "HongxinLi/UIPro-7B_Stage2_Mobile", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("HongxinLi/UIPro-7B_Stage2_Mobile")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./web_6f93090a-81f6-489e-bb35-1a2838b18c01.png",
            },

            {"type": "text", "text": """Given the Mobile UI screenshot and previous actions, please generate the next move necessary to advance towards task completion. The user's task is: {task}
Action history: {action_history}

Now, first describe the action intent and then directly plan the next action."""},
        ],
    }
]

```

---

<div align="center">

## 🧪 **Reproducing Agentic Task Evaluation**

*Run the agentic task evaluation scripts for AiTW / AndroidControl / GUIAct / Mind2Web.*

</div>

> **Note**
> - These scripts perform **online action prediction** step-by-step (i.e., they will run model inference), so a GPU environment is recommended.
> - Outputs are saved as JSON files under `utils/eval_utils/eval_results/`.

#### 🤗 Evaluation Datasets on HuggingFace

The test annotations loaded by the four evaluation scripts are released publicly:

| Benchmark | Script | HuggingFace |
| :--- | :--- | :--- |
| AiTW | `eval_aitw.py` | [AITW_test](https://huggingface.co/datasets/HongxinLi/AITW_test) |
| AndroidControl | `eval_androidcontrol.py` | [AndroidControl_test](https://huggingface.co/datasets/HongxinLi/AndroidControl_test) |
| GUIAct | `eval_guiact.py` | [GUIAct](https://huggingface.co/datasets/HongxinLi/GUIAct) |
| Mind2Web | `eval_mind2web.py` | [Mind2Web_test](https://huggingface.co/datasets/HongxinLi/Mind2Web_test) |


<details>
<summary><b>🤖 AiTW (Android in the Wild) — `utils/eval_utils/eval_aitw.py`</b></summary>
<br>

**What you need**
- **AiTW screenshots** (see the *Agentic SFT Data Processing → Android in the Wild (AiTW)* section below). You only need the image folder for evaluation.
- The script loads the **test annotations** from HuggingFace: `HongxinLi/AITW_test`.

**Expected images layout**
```
/path/to/AITW/aitw_images/
├── general/
├── googleapps/
├── install/
├── single/
└── webshopping/
```

**Run**

```bash
python utils/eval_utils/eval_aitw.py \
  --pretrained "HongxinLi/UIPro-7B_Stage2_Mobile" \
  --imgs_dir "/path/to/AITW/aitw_images" \
  --action_refexp \
  --scale 1000
```

**Common options**
- `--debug`: only evaluate a few steps for quick sanity check
- `--cot`: enable Chain-of-Thought prompting
- `--action_refexp`: used to drive the UIPro model to output a short action intent before predicting the action function call.
- `--max_prev_acts`: how many previous actions are included in history
- `--original_actspace`: use the original AITW action space in prompts. Ignore this to use the unified action space provided by UIPro.

**Outputs**
- Saved to `utils/eval_utils/eval_results/AITW/<model_postfix>[_CoT][_wActRef]/<timestamp>.json`

</details>

<details>
<summary><b>🎮 AndroidControl — `utils/eval_utils/eval_androidcontrol.py`</b></summary>
<br>

**What you need**
- A local AndroidControl test JSON (e.g., the processed file produced by `utils/data_utils/make_androidcontrol_data/make_androidcontrol_data.py`).
- The images should be reachable by joining `--data_root` with the `image` field in the test JSON.

We also provide a preprocessed zip file of the AndroidControl test data used in the UIPro paper. First download and unzip the GoClick AndroidControl Test Data via
`hf download HongxinLi/AndroidControl_test  --repo-type dataset --local-dir path/to/AndroidControl_test`

**Run**
```bash
python utils/eval_utils/eval_androidcontrol.py \
  --pretrained "HongxinLi/UIPro-7B_Stage2_Mobile" \
  --testset_path "utils/eval_utils/AndroidControl-test_12685.json" \
  --data_root "/path/to/AndroidControl"
```

**Useful options**
- `--action_refexp`: (only used for UIPro) Please use this to prompt UIPro to output short reasoning before planning the action 

**Outputs**
- Saved to `utils/eval_utils/eval_results/androidcontrol/<model_postfix>/<timestamp>.json`

</details>

<details>
<summary><b>🎭 GUIAct — `utils/eval_utils/eval_guiact.py`</b></summary>
<br>

**What you need**
- A local GUIAct processed folder that contains the test JSON and images referenced by the `image` field.
- You can pass `--root` to point to your processed GUIAct directory, and `--imgs_dir` to override the image base.
  If omitted, the script auto-detects local/server paths inside `get_dataset_paths()` in `utils/eval_utils/eval_guiact.py`.

We also provide the preprocessed GUIAct test data zip file. First download the GUIAct data from HongxinLi/GUIAct, unzip it, and organize it as follows:
```
root/
├── GUICourse/
│   └── GUIAct/
│   │    └── imgs
│   ├── Web_test.json
│   └── Mobile_test.json
```


**Run**
```bash
python utils/eval_utils/eval_guiact.py \
  --pretrained "HongxinLi/UIPro-7B_Stage2_Web" \
  --device_type Web \
  --scale 1000 \
  --root "/path/to/GUICourse_processed" \
  --imgs_dir "/path/to/images_base"
```

**Common options**
- `--device_type`: `Web` or `Mobile`
- `--root`: processed GUIAct directory (overrides auto-detect)
- `--imgs_dir`: image base directory (overrides auto-detect)
- `--debug`: evaluate a small random subset
- `--cot`: enable Chain-of-Thought prompting
- `--action_refexp`: (only used for UIPro) Please use this to prompt UIPro to output short reasoning before planning the action 
- `--original_actspace`: use the original GUIAct action space

**Outputs**
- Saved to `utils/eval_utils/eval_results/GUIAct-<device_type>/<model_postfix>/<timestamp>.json`

</details>

<details>
<summary><b>🌐 Mind2Web — `utils/eval_utils/eval_mind2web.py`</b></summary>
<br>

**What you need**
- A local Mind2Web folder that contains:
  - `mind2web_images/` (evaluation screenshots)
  - `mind2web_data_test_{website|task|domain}.json` (test annotations)
- You can pass `--root` to point to your local Mind2Web directory. If omitted, the script auto-selects
  the first existing path in `DEFAULT_ROOTS` inside `utils/eval_utils/eval_mind2web.py`.

**Expected layout**
```
/path/to/Mind2Web/
├── mind2web_images/
└── mind2web_data_test_website.json
    mind2web_data_test_task.json
    mind2web_data_test_domain.json
```

**Run**
```bash
python utils/eval_utils/eval_mind2web.py \
  --pretrained "HongxinLi/UIPro-7B_Stage2_Web" \
  --task website \
  --scale 1000 \
  --root "/path/to/Mind2Web"
```

**Common options**
- `--task`: either 'task', 'domain', or 'website' (the three splits of Mind2Web)
- `--debug`: only evaluate a few episodes
- `--cot`: enable Chain-of-Thought prompting
- `--action_refexp`: (only used for UIPro) Please use this to prompt UIPro to output short reasoning before planning the action 
- `--max_prev_acts`: how many previous actions are included in history

**Outputs**
- Saved to `utils/eval_utils/eval_results/mind2web/<model_postfix>/{website|task|domain}-<timestamp>.json`

</details>

<div align="center">

---

<div align="center">

## 📝 **Citation**

*If you use UIPro in your research, please cite our paper*

</div>

```bibtex
@inproceedings{li2025uipro,
  title={UIPro: Unleashing Superior Interaction Capability For GUI Agents},
  author={Li, Hongxin and Su, Jingran and Chen, Jingfan and Ju, Zheng and Chen, Yuntao and Li, Qing and Zhang, Zhaoxiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

---

<div align="center">

## 👥 **Team & Acknowledgments**

*Special thanks to our research team and the open-source community*

<p>
This work was supported in part by the <b>National Key R&D Program of China</b> and the <b>National Natural Science Foundation of China</b>. We extend our gratitude to the open-source community for providing foundational datasets and tools that made this research possible.
</p>

<br>

---

<br>

## ⭐ **Star this repository if you find UIPro helpful!** ⭐

<a href="https://github.com/ZJULiHongxin/UIPro/stargazers">
  <img src="https://img.shields.io/github/stars/ZJULiHongxin/UIPro?style=for-the-badge&logo=github&logoColor=white&labelColor=1A1A2E&color=FFD93D" alt="GitHub Stars"/>
</a>

<br><br>

<p><i>🚀 Revolutionizing GUI automation, one interaction at a time</i></p>

</div>
