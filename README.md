# 人体合成数据生成器

这个项目用已有的人物和动作资源，生成走路、骑行、多人经过等场景，供 MAMMA 训练使用。

你可以分别选择**人物身体、动作、衣物和发型**，设置路线与相机位置。也可以给出抽样范围，让程序自动生成不同的场景配置。交互式编辑器用于查看和调整场景，批量生成不需要逐个手动摆放。

每个相机输出：

- RGB 图片和每个人的 mask。
- 每个人有序的 512 点二维坐标。
- 每个点的可见性：`1` 可见、`0` 被遮挡、`-1` 无法确定。

场景中每个人都作为一个训练目标。框由该人物的 mask 得到；没有可见 mask 的人物仍记录为 `absent`。本项目不生成接触标签。

## 本轮交付状态（2026-10-25）

已完成原城市20个及新增遮挡15个完整数据包，MAMMA按28个场景训练、7个场景验证。七份实际用户规则位于 `outputs/rules/`；下文 `config/batch_rules.json` 是通用示例，不等于本轮七份规则。

当前代码与使用入口在本次收尾固定；原始数据、几何缓存、规则和渲染输出仍在本机，不纳入Git。规则校验值、训练最终权重、真实合图效果与运行入口见[本轮联合交付说明](https://github.com/casesic/mamma_retrain/blob/main/docs/DELIVERY_20261025.md)。新增遮挡数据记录见[15场景报告](docs/OCCLUSION_RULES_FIVE_RENDER_20261025.md)。

## 外部依赖：mamma_retrain

本仓库只产出数据包（`observations.json` + 每台相机的 RGB / 人物 mask / 有序 512 点标签），**训练和评价在另一个仓库**：

- **是什么**：`mamma_retrain` 是 MAMMA 的 fork，读这份数据包做微调、可见性阈值研究和定量评价。本仓的产出就是它的训练输入。
- **在哪**：<https://github.com/casesic/mamma_retrain>（公开）。
- **怎么获取**：`git clone https://github.com/casesic/mamma_retrain.git`，放到与本仓**同一父目录**——历史文档里写成 `../mamma_retrain/...` 就是这个约定。
- **为什么不写进依赖**：本仓**不 import 它**，两边只通过文件格式交接（数据包清单 + observations），所以它不在 `pyproject.toml` 里。
- **另需一个资源文件**：`config/resources.json` 的 `mamma_template` 默认指向 `mamma_retrain/data/body_models/downsampled_verts/verts_512.pkl`；该权重需在 <https://mamma.is.tue.mpg.de/> 注册后下载，见 [DATA_SOURCES](DATA_SOURCES.md)。
- **四仓交接**：[PIPELINE_CONTRACT](https://github.com/casesic/mamma_retrain/blob/main/docs/PIPELINE_CONTRACT.md) 写清了本仓产出的数据包被谁消费、字段定义在哪。
- **一处已失效的旧引用**：早前文档让读者去改 `mamma_retrain/configs/retrain/mamma_saved_gt_reader.json`，这个文件**在 mamma_retrain 里已经不存在**；训练端现在用的是 `configs/retrain/mamma_regular.json` 一类配置，以它的[当前用法](https://github.com/casesic/mamma_retrain/blob/main/docs/MAMMA_CURRENT_ENTRY.md)为准。

## 1. 开始使用

以下命令在项目目录执行，使用本机已有的 Python 环境，无需重新安装本包：

```bash
cd <本仓库目录>
export PYTHONPATH="$PWD/src"
GEN_PY=<你的 Python>          # 需能 import OpenEXR
```

每次新开终端，先执行上面这三行。`GEN_PY` 会同时成为 CPU consumer 的解释器，所以它必须读得了 OpenEXR —— 当时的 conda 环境里 `densekp_render` 通过真实渲染验证、`camerahmr` 缺 `OpenEXR`。换机器时改 `GEN_PY` 与 [resources.json](config/resources.json) 里的资源位置即可。

常用文件：

| 想做什么 | 看哪个文件 |
|---|---|
| 设置批量抽样规则 | [config/batch_rules.json](config/batch_rules.json) |
| 手动配置一个确定的场景 | [config/scene_example.json](config/scene_example.json) |
| 查看人物、动作、衣物、发型资源 | [config/catalog/](config/catalog/) |
| 设置资源、Blender、标定和模板的位置 | [config/resources.json](config/resources.json) |

## 2. 详细流程

从「按规则生成一批场景」到「常见情况与当前验证范围」，完整六节在 **[docs/USAGE.md](docs/USAGE.md)**：

1. 推荐流程：生成场景配置 → 检查场景 → 准备几何与相机裁剪 → 渲染并导出
2. 怎样设置随机规则
3. 只做单个场景或预览
4. 使用交互式编辑器
5. 生成结果在哪里（含交给 MAMMA 读取）
6. 常见情况与当前验证范围

