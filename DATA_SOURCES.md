# 数据来源与重建方式

本仓库只包含生成器源码、配置与文档。人物/动作/衣物来源、几何缓存、规则与
渲染输出均在本机，按 `.gitignore` 不纳入版本控制。

## 不在仓库里的内容

| 路径 | 体积 | 内容 | 取得方式 |
| --- | --- | --- | --- |
| `outputs/` | 5.4 GB | 渲染图、mask、512 点标签、规则、几何缓存 | 由本仓库重新生成，见下文「重建」 |
| `assets/`、`data/`、`checkpoints/` | — | 本地人物资产、动作片段、模型权重（当前机器上不存在） | 见下方依赖 |
| `outputs/rules/` | — | 本轮七份用户规则（业务输入，非代码） | 需从交付沟通中重新确认，无法从上游下载 |

## 上游依赖

生成器本身不含人物与动作，运行时从下列位置读取：

1. **MAMMA 512 顶点模板** — `config/resources.json` 里的 `mamma_template`，
   默认指向 `mamma_retrain/data/body_models/downsampled_verts/verts_512.pkl`。
   取得方式见 [mamma_retrain/DATA_SOURCES.md](../mamma_retrain/DATA_SOURCES.md)：
   `bash data/download_mamma_weights.sh --verts`（需在 `https://mamma.is.tue.mpg.de/` 注册）。
2. **场景与相机标定** — `config/environments/*.json` 与
   `config/resources.json` 的 `calibration_dir`，指向
   `densekp_retrain/outputs/experiments/t_junction_multirig_a1_20261011/prepared_scene/calibration`。
   该目录随 `densekp_retrain/outputs/` 一起不在仓库内，需按
   [densekp_retrain/DATA_SOURCES.md](../densekp_retrain/DATA_SOURCES.md) 重建。
3. **Blender** — `config/resources.json` 里 `blender_executable` 写死为本机路径，
   换机后必须修改。Blender 下载：`https://www.blender.org/download/`。
4. **SMPL-X 人体模型** — `https://smpl-x.is.tue.mpg.de/`，需单独申请授权，
   条款不允许转发模型权重。

## 重建

```bash
# 1. 准备环境
pip install -e .

# 2. 修正 config/resources.json 中的三处本机绝对路径：
#    mamma_template / blender_executable / calibration_dir

# 3. 生成（示例）
python -m human_data_generator.cli --help
```

`config/resources.json` 里是**绝对路径**，换机器后不会自动生效，
这是本项目迁移时最容易踩的坑。

## 与训练的关系

本轮交付的 35 个场景（原城市 20 + 新增遮挡 15）按 28 训练 / 7 验证划分，
训练侧说明见 `mamma_retrain/docs/DELIVERY_20261025.md`。
训练与评估结果不在本仓库，也不在 mamma_retrain 仓库内。
