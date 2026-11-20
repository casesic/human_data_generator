# AGENTS.md

给在本仓库改代码的人（和自动化代理）的约定。读代码的路线另见 [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)，使用手册见 [README.md](README.md)。

## 这个仓库做什么，不做什么

用现成的人物 / 动作 / 衣物 / 发型资产，在 Blender 里渲染出「走路、骑行、多人经过」等场景，输出**每台相机的 RGB、目标人物 mask、有序 512 点坐标与可见性**，最后汇总成 `observations.json` 数据包。

- **做**：场景配置 → 计划 → 渲染 → CPU 端整理 → 导出数据包。
- **不做**：训练、评价、阈值研究。那些在 `mamma_retrain`（外部仓库，见 README 的「外部依赖」）。
- **不做**：检测框标注。框由目标人物 mask 在训练端算出来。

## 技术栈

| 层 | 用什么 | 说明 |
|---|---|---|
| CLI 与计划 | Python 3.10+，stdlib + numpy | 包在 `src/human_data_generator/`，可 editable 安装，也可直接 `PYTHONPATH=src` |
| 渲染 | Blender 自带 Python（`bpy`、`mathutils`） | **不是 pip 依赖**，只能用 Blender 提供的解释器 |
| CPU 端整理 | numpy、Pillow、OpenCV（只用到 `cv2.boundingRect`）、joblib、OpenEXR | 见 `pyproject.toml` 的 `[project] dependencies` |
| 可选 | `av`（视频组装）、`torch` + `smplx`（body-rest 组合） | 惰性 import，装在 `[video]` / `[fitting]` extra 里 |

## 目录职责

| 路径 | 负责 |
|---|---|
| `run.py` | CLI：`plan` / `prepare` / `render` / `run-prepared` / `export` 五个子命令 |
| `batch.py` | 按 `config/batch_rules.json` 抽样出场景 JSON（批量到此结束，不重新抽） |
| `components.py` | 人物 × 动作 × 衣物 × 发型能不能组合的合法性检查 |
| `composition.py` | 构建人物几何（body-rest 路径会惰性 import torch/smplx） |
| `runtime/scene.py` | 场景配置校验；`runtime/multirig_prepare.py` 出逐帧/逐相机计划 |
| `runtime/frame_consumer.py` | 同步子进程客户端，**只允许 stdlib**（要在 Blender 里 import） |
| `render/` | Blender 侧装配（`assembly.py`）与相机循环（`multirig.py`） |
| `rendering/` | 材质、groom、Blender 资产组装 |
| `consumer.py` | CPU 端把渲染缓冲整理成每台相机的文件 |
| `data/` | `mamma_512` 标签、mask 派生框、`observations.json` 导出 |
| `io/`、`imaging/` | EXR 通道读取、图像来源 |
| `editor/` | 浏览器编辑器页面与本地 HTTP 服务 |
| `config/`、`docs/`、`tests/` | 配置示例与规则、文档、CPU 测试 |

## 命名与格式（改了会破坏下游）

- 人物编号 `person_id` 是整数；相机名是 `cam2` / `cam4`；帧目录 `frame_000`。
- 每台相机一个目录：`rigs/<rig>/frame_XXX/<cam>/`，内含 `roi.png`、`person_<id>_mask.png`、`mamma_512.npz`、`annotations.npz`、`raster_storage.json`。
- **512 点的 XY 是完整 ERP 坐标**，`roi.png` 只是局部图；局部图在全景中的位置由 `raster_storage.json` 说明。不要把两者当成同一张坐标系。
- `observations.json` 的一条记录 = 一个（帧, 相机, 人）。没有可见 mask 的人仍要记一条 `absent`，不要静默丢人。
- 可见性三态：`1` 可见、`0` 被遮挡、`-1` 无法确定。`-1` 不取消坐标监督。

## 禁止事项

1. **不要改** `observations.json` / `mamma_512.npz` 既有字段的语义——`mamma_retrain` 按这些字段读，改了等于悄悄换格式。
2. **不要**把本机绝对路径写进仓库（文档、配置示例、默认值都算）。
3. **不要**绕过 `export` 的覆盖保护：已有 `observations.json` 时应当失败，不要加 `--force` 之类的口子。
4. **不要**删除或改写 `outputs/` 下的历史产物。它们不在 Git 里，可能只有一份。
5. **不要**在没有显式证据时把 `-1` 改成 `0`/`1`。遮挡规则的做法与边界见 `docs/OCCLUSION_RULES_*.md`。
6. **不要**在 `render/`、`rendering/` 之外的 Blender 运行路径里 import 重依赖；`runtime/frame_consumer.py` 刻意只用 stdlib。
7. **不要**在这条 CPU 路径里引入新的第三方库而不更新 `pyproject.toml`。

## 改完必须跑

```bash
PYTHONPATH=src python tests/test_current_pipeline.py
```

这是唯一的 CPU 测试文件（`unittest`），覆盖 batch 抽样、计划、导出与 mask 缺失等边界。前提是装好 `numpy` / `Pillow` / `opencv` / `joblib`；缺 `cv2` 会直接在 import 阶段失败（`data/mamma_crop.py` 顶层 import 它）。

改动 `render/`、`rendering/` 或 `runtime/frame_consumer.py` 时，CPU 测试**覆盖不到** `bpy`：必须另外在 Blender 里跑一次 `prepare` + `render`，并说明用的 GPU 与帧号。

## 提交约定

- 一次提交只做一件事；改动行为与改文档分开。
- 提交信息用英文祈使句（与历史一致），正文写清「为什么」，不要只写「update」。
- 渲染产物、数据包、权重都不进 Git；需要留证据就写进 `docs/` 报告并链接文件路径。
