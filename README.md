# PSD 自动化标注
 Live2D/插画 PSD 文件自动图层标注。

## 1. 项目目标
- 批量处理指定目录下的 PSD 文件。
- 每个 PSD 自动导出整图 PNG 与逐层 PNG。
- 将"整图 + 单层 + 图层元信息 + 命名规则"提交给 Gemini，多模态生成 `PSD：原名 -> 新名` 的映射。
- 结果写入 JSON/CSV，完成后清理临时 PNG。
- 支持多 API key、限速与并发配置，可扩展大批量处理。
  
 ## ⚠️ 重要提示

**此项目会消耗大量 API 请求！**

- 每个图层需要 1 次 Gemini API 调用（包含 2 张图片 + 文本提示）
- 一个典型的 Live2D PSD 文件包含 **100-300 个图层**
- 示例：150 图层的 PSD = 150 次 API 请求

**建议**：
1. 🧪 先用 `--limit 1` 测试单个小文件
2. 💰 检查你的 Gemini API 配额和计费
3. 🔑 配置多个 API Key 分散负载
4. ⏱️ 合理设置 `rpm`（每分钟请求数）避免超限

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置 API Key**

   创建 `.env` 文件并添加你的 Gemini API Keys：
   ```bash
   GEMINI_KEY_1=your_first_api_key
   GEMINI_KEY_2=your_second_api_key
   # 可添加更多...
   ```

3. **放入 PSD 文件**

   将待处理的 PSD 文件放入 `psd/` 目录。

4. **运行处理**
   ```bash
   python scripts/dispatcher.py --config config/settings.yaml

   # 或限制处理数量
   python scripts/dispatcher.py --config config/settings.yaml --limit 1
   ```

5. **查看结果**

   处理完成后在 `output/results/` 目录查看 JSON 结果文件。

## 2. 特性

✅ **自动编码修复**：自动修复 Live2D/Photoshop PSD 文件的 Mac Roman 编码问题，正确显示中文组名
✅ **多 API Key 支持**：支持配置多个 Gemini API Key，自动轮询和故障转移
✅ **智能重试机制**：遇到 503/429 错误时自动重新排队并冷却
✅ **并发处理**：可配置 PSD 级和 API Key 级并发数
✅ **速率限制**：支持设置每分钟请求数（RPM）防止超限
✅ **大 Token 输出**：默认 6400 tokens，适应复杂 PSD 的大量图层

## 3. 目录结构

```
project/
├─ config/
│  ├─ settings.yaml          # 配置
│  └─ prompts/rename.txt     # 提示词模板
├─ psd/                      # 待处理 PSD
├─ tmp/                      # 导出的 PNG（按 PSD 建子目录）
├─ output/
│  ├─ results/               # 每 PSD 的结果 JSON/CSV
│  └─ logs/                  # 日志
└─ scripts/
   ├─ process_psd.py         # 导入/导出 PSD
   ├─ gemini_client.py       # Gemini SDK 封装
   ├─ dispatcher.py          # 调度器+并发
   └─ cleanup.py             # 清理工具
```

## 4. 配置 (`config/settings.yaml`)
- `gemini`: 模型名、温度、top_k、top_p、max_output_tokens（默认 6400，适应复杂 PSD 的大量图层）。
- `api_keys`: 数组；每个 key 定义 `key`（可写 `${ENV_NAME}` 让脚本从环境变量读取）、`max_concurrency`（该 key 可同时发起的请求数）、`rpm`（每分钟允许的模型调用次数，用于节流；若触发 429/503 会自动冷却 `GEMINI_QUOTA_COOLDOWN` 秒后重试其他 key）。
- 如果需要自定义 Gemini 端点（例如本地代理），可在 `.env` 中写 `GEMINI_ENDPOINT=http://127.0.0.1:8003`，脚本会改用 REST 接口走该域；否则默认使用官方 google-genai SDK。
- `processing`:  
  - `include_hidden_layers`, `flatten_groups`, `max_psd_concurrent`（同时处理的 PSD 数量）、  
    `png_format`, `png_dpi`, `retry`（`max_attempts`, `backoff_seconds` 控制单层调用失败后的重试策略）。
- `prompt`:  
  - `template_path`, `output_format`（固定一行 `PSD：原名 -> 新名`）。
- `paths`: `psd_dir`, `tmp_dir`, `results_dir`, `log_file`.
- `logging`: 等级、日志输出路径。

## 5. 提示词模板 (`config/prompts/rename.txt`)
- 包含命名规则、左右区分要求、父组语义提示等。
- 占位符：`{psd_name}`, `{original_name}`, `{group_path}`, `{layer_metadata}`, `{merged_png}`, `{layer_png}`。
- 输出要求：仅返回 `PSD：原名 -> 新名` 一行。


## 6. Gemini SDK 使用说明（模型 gemini-2.5-pro）
- 参考 [Gemini API quickstart](https://ai.google.dev/gemini-api/docs/quickstart) 安装官方 `google-genai` SDK（Python ≥3.9）。
- 安装依赖：
  ```bash
  pip install -q -U google-genai
  ```
- 设置 API Key：
  - Linux/macOS：`export GEMINI_API_KEY=\"<your-key>\"`
  - Windows（PowerShell）：`setx GEMINI_API_KEY \"<your-key>\"`
- SDK 会自动从环境变量读取 key，也可以在 `genai.Client(api_key=...)` 中显式传入。
- 所有请求统一使用 `model=\"gemini-2.5-pro\"`，满足高精度需求。
- 多模态输入可参考 [Vision 指南](https://ai.google.dev/gemini-api/docs/vision)，用 `types.Part.from_bytes(data=..., mime_type=\"image/png\")` 附加 PNG，并配合文本提示描述上下文。

### 6.1 Python 多模态示例
```python
import os
from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get(\"GEMINI_API_KEY\"))

def rename_layer(psd_name, original_name, prompt_text, merged_png, layer_png):
    merged_bytes = Path(merged_png).read_bytes()
    layer_bytes = Path(layer_png).read_bytes()

    response = client.models.generate_content(
        model=\"gemini-2.5-pro\",
        contents=[
            types.Part.from_text(prompt_text),
            types.Part.from_bytes(data=merged_bytes, mime_type=\"image/png\"),
            types.Part.from_bytes(data=layer_bytes, mime_type=\"image/png\"),
            f"{psd_name}：{original_name}"
        ],
    )
    return response.text.strip()
```
该函数示例展示了如何把提示词、整图 PNG、单图层 PNG 与“PSD：原名”上下文一并提交给 Gemini 2.5 Pro，返回值即可写入结果表。


## 7. 模块说明

### 7.1 `process_psd.py`
1. `PSDImage.open(path)`。
2. 导出整图 `merged.png` (`psd.composite()` 或 `psd.topil()`).
3. 遍历 `psd.descendants()`：
   - 根据配置跳过组/隐藏层。
   - `layer.topil().save(...)` → `tmp/<psd>/<index>_<safe_name>.png`。
   - 记录 `LayerTask`：原名、父级路径、可见性、位置尺寸、PNG 路径。
4. 返回层任务列表 + 合成图路径。

### 7.2 `gemini_client.py`
- 封装 Gemini SDK：
  - `configure(api_key)`.
  - `generate_name(task, prompt_template)`：
    - 读取整图/层图 → `GenerativeModel.generate_content([prompt, image0, image1])`。
    - 解析输出，只保留一行映射；失败抛异常供调度器重试。

### 7.3 `dispatcher.py`
1. 解析配置，扫描 `psd_dir` → `PSDJob` 队列。
2. 控制 `max_psd_concurrent` 个 PSD 同时运行。
3. 对每个 PSD：
   - 调 `process_psd` 得到 `LayerTask` 列表。
   - 放入 `asyncio.Queue`（任务包含 PSD 名、层信息、PNG 路径）。
4. 为每个 API key 创建 `max_concurrency` 个 worker：
   - Worker 固定使用对应 key，遵守 `rpm`（基于下一次可用时间）。
   - 从队列取任务 → 调 `gemini_client` → 写入临时结果。
   - 失败时按 `retry` 策略重试，超限标记失败。
5. 当 PSD 所有任务完成：
   - 输出 `output/results/<psd>.json`（或 CSV）。
   - 调 `cleanup_psd` 删除 `tmp/<psd>/`。
6. CLI 支持 `--psd-limit`, `--resume failed` 等参数。

### 7.4 `cleanup.py`
- `cleanup_psd(psd_name)`: 删除 `tmp/<psd_name>/`。
- `cleanup_stale(hours)`: 启动时/定期清理残留目录。

## 8. 输出与日志
- JSON 结构：
  ```json
  {
    "psd": "free1.model3.psd",
    "composite": "E:\\path\\to\\tmp\\free1.model3\\free1.model3_composite.png",
    "layers": [
      {
        "index": 1,
        "original": "ArtMesh172",
        "group_path": "Root/資料夾 2/后头发 的複製/右发带 的複製",
        "layer_png": "E:\\path\\to\\tmp\\free1.model3\\layer_0001_ArtMesh172.png",
        "output": "free1.model3.psd：ArtMesh172 -> 裙子前",
        "status": "ok",
        "attempts": 1
      }
    ]
  }
  ```
- `group_path` 自动修复了 Live2D/Photoshop 的编码问题（Mac Roman → GBK），繁体中文组名可正确显示。
- CSV（可选）：`PSD,Original,New,Status`.
- 日志写入 `output/logs/dispatcher.log`，记录任务开始、完成、重试。

## 9. 并发与速率控制
- PSD 并发：`max_psd_concurrent` 限制同时处理的 PSD 数。
- API key 并发：`max_concurrency` 控制每个 key 同时请求数。
- 速率：`rpm` → 计算 `next_available_time`；worker 等待到时间再发送。
- 支持多 key 轮询；单 key 多并发（如 `max_concurrency=2` 实现“一 key 两并发”）。

## 10. 运行流程
1. 把 PSD 放入 `psd/`；填好 `settings.yaml`、`prompts/rename.txt`。
2. `pip install -r requirements.txt`
3. `python scripts/dispatcher.py --config config/settings.yaml`
3. 处理完成后在 `output/results/` 查看命名结果。
4. 若需重跑失败任务，用 `--resume failed`。
5. 定期运行 `cleanup_stale` 清理 tmp。


## 11. 后续扩展
- 导出层描述、部件摘要等附加信息。
- Web UI / Dashboard 查看进度。
- Docker 化或结合任务队列（Celery/RQ）。
- 写集成测试，mock Gemini 接口。
- 支持更多编码格式（Shift-JIS、Big5 等）。
