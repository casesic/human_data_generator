# 人体合成数据生成器

这个项目用已有的人物和动作资源，生成走路、骑行、多人经过等场景，供 MAMMA 训练使用。

你可以分别选择**人物身体、动作、衣物和发型**，设置路线与相机位置。也可以给出抽样范围，让程序自动生成不同的场景配置。交互式编辑器用于查看和调整场景，批量生成不需要逐个手动摆放。

每个相机输出：

- RGB 图片和每个人的 mask。
- 每个人有序的 512 点二维坐标。
- 每个点的可见性：`1` 可见、`0` 被遮挡、`-1` 无法确定。

场景中每个人都作为一个训练目标。框由该人物的 mask 得到；没有可见 mask 的人物仍记录为 `absent`。本项目不生成接触标签。

## 1. 开始使用

以下命令在项目目录执行，使用本机已有的 Python 环境，无需重新安装本包：

```bash
cd /mnt/data1/yy/home/yy/Projects/human_data_generator
export PYTHONPATH="$PWD/src"
GEN_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python
```

每次新开终端，先执行上面这三行。换到其他机器时，需要调整 Python 路径和 [resources.json](config/resources.json) 中的资源位置。

常用文件：

| 想做什么 | 看哪个文件 |
|---|---|
| 设置批量抽样规则 | [config/batch_rules.json](config/batch_rules.json) |
| 手动配置一个确定的场景 | [config/scene_example.json](config/scene_example.json) |
| 查看人物、动作、衣物、发型资源 | [config/catalog/](config/catalog/) |
| 设置资源、Blender、标定和模板的位置 | [config/resources.json](config/resources.json) |

## 2. 推荐流程：按规则生成一批场景

流程分成四步：**抽样配置 → 检查 → 准备几何 → 渲染并导出数据**。

### 第一步：生成场景配置

修改 [batch_rules.json](config/batch_rules.json)，然后运行：

```bash
"$GEN_PY" -m human_data_generator.batch \
  --rules config/batch_rules.json \
  --name key_batch_001
```

结果放在 `outputs/scenes/key_batch_001/`。每份场景 JSON 都已经确定了人物、动作、衣物、发型、路线和机位；`batch.json` 列出这批场景文件的位置。

**这一步只写 JSON，不构建几何、不渲染。**当前批量入口也不会自动执行后面的准备和渲染。

默认规则有城市单人步行、城市单人骑行、城市双人相向经过、河岸单人骑行。双人规则当前两人都使用骑行动作。

### 第二步：检查一份场景

以默认规则生成的第一份场景为例：

```bash
SCENE=outputs/scenes/key_batch_001/batch_000_city_single_walk.json

"$GEN_PY" -m human_data_generator.run plan --scene "$SCENE"
```

`plan` 只读检查配置、资源和动作范围，不构建缓存，也不使用 GPU。修改规则后，建议先检查再准备。

### 第三步：准备几何和相机裁剪范围

```bash
"$GEN_PY" -m human_data_generator.run prepare \
  --scene "$SCENE" \
  --name walk_001
```

这一步在 CPU 上按人物选择准备身体、衣物和发型几何，并计算相机需要渲染的区域。已有匹配的组合缓存会复用。

准备结果放在 `outputs/runs/walk_001/`。**此时还没有 RGB 图片。**任务名称应使用新的名称，程序不会覆盖已有的非空任务目录。

### 第四步：渲染并导出数据

下面的命令会真正启动 Blender 和 GPU。将 `GPU-实际UUID` 换成要使用的 GPU UUID：

```bash
"$GEN_PY" -m human_data_generator.run run-prepared \
  --root outputs/runs/walk_001 \
  --gpu-uuid GPU-实际UUID \
  --budget-seconds 1800
```

`1800` 表示本次执行最多使用 1800 秒，即 30 分钟；计时从执行时开始。成功后会自动整理图片、标签并生成 `observations.json`，不需要再手动导出。

也可以把准备和渲染合成一个命令，使用另一个新任务名：

```bash
"$GEN_PY" -m human_data_generator.run render \
  --scene "$SCENE" \
  --name walk_002 \
  --gpu-uuid GPU-实际UUID \
  --budget-seconds 1800
```

需要展示视频时，在渲染命令末尾加 `--video`。视频在数据导出后制作，视频制作失败不会删除已生成的数据。

## 3. 怎样修改批量规则

`batch_rules.json` 的 `scenes` 中，每一项生成一个场景。复制一项、修改内容并给它新的 `scene_id`，就能增加一项规则，不需要修改 Python。

### 路线、速度与相机

| 字段 | 含义 |
|---|---|
| `environment` | 环境，当前为 `city` 或 `riverfront` |
| `fps` / `frame_count` | 输出帧率 / 帧数 |
| `route_start_xy_range_m` | 路线起点的 x、y 抽样范围，单位为米 |
| `route_end_xy_range_m` | 路线终点的 x、y 抽样范围 |
| `speed_m_s` | 沿路线运动的速度范围，单位为米/秒 |
| `start_progress_m` | 开始时已沿路线前进的距离范围 |
| `motion_start` | 源动作起始帧的抽样范围，按动作时间线索引选择，不是输出帧号 |
| `camera_jitter.xy_m` | 相机组 x、y 各自允许的最大随机偏移，单位为米 |
| `camera_jitter.yaw_deg` | 相机组朝向允许的最大随机偏移，单位为度 |

例如：

```json
"route_start_xy_range_m": [[-1, 1], [77.96, 77.96]],
"speed_m_s": [0.8, 1.3],
"motion_start": [0, 4]
```

表示起点 x 在 −1 到 1 米之间抽取，y 固定为 77.96 米；速度在 0.8 到 1.3 米/秒之间抽取；动作从第 0 到第 4 帧中选一个起点。范围上下界相同表示固定值。

`rigs` 是相机组列表，每组包含 `cam2`、`cam4` 两个相机。`origin_m` 与 `yaw_deg` 设置整组的位置和朝向，组内相对标定保持不变。

相同规则、资源、总 seed 和场景在列表中的编号会得到相同结果。调整 `scenes` 的顺序会改变编号，因而可能改变抽样结果。

### 人物、动作、衣物和发型

每个人有自己的 `components` 候选范围。例如：

```json
"components": {
  "body": {
    "default_versions_only": true,
    "ids": ["M01", "M03"]
  },
  "motion": {
    "behavior": "walk",
    "ids": ["KIT425_walk", "KIT11_walk"]
  },
  "clothing": {
    "kinds": ["simple_barycentric"],
    "ids": ["M01_simple", "M03_simple"]
  },
  "hair": {
    "ids": ["M01_groom", "M03_groom"]
  }
}
```

这个例子从指定的两个人物、两条走路动作、两件衣物和两个发型中选择组合。

- `ids: []`：不限定具体 ID，从满足其他条件的全部可用资源中选。
- `ids: ["M01"]`：固定选择这个 ID。
- `default_versions_only: true`：身体只选各身份的默认版本。要使用其他可用版本，例如 `F04_v1`，设为 `false` 并指定对应 ID。
- `behavior`：`walk` 为走路，`ride` 为骑行。
- `simple_barycentric`：可按目标身体重新绑定的普通衣物。
- `native_locked`：只能用于特定身体、动作和时间段的已有衣物缓存。

所有条件同时生效。指定不存在、不可用或不满足条件的 ID 会报错，不会偷偷换成其他资源。候选 ID 列表换序不会改变抽样结果。

现有 `M03_KIT11_BEDLAM_transfer` 衣物需要搭配 `M03` 身体、`KIT11_walk` 动作，选中的动作帧必须在 `[0,120)` 内；发型可独立选择。把它加入规则时，同时设置 `clothing.kinds: ["native_locked"]` 和对应 ID。

查看实际可用 ID、动作帧数及限制：

```bash
"$GEN_PY" -m human_data_generator.components
"$GEN_PY" -m human_data_generator.components --kind motions
```

`--kind` 还可以用 `bodies`、`clothing`、`hair`。目录也列出了暂不可用的资源及原因，暂不可用项不能直接参与生成。

## 4. 只做单个场景或预览

想精确指定每个人的组合和路线，可以直接修改 [scene_example.json](config/scene_example.json)。它保存的是已经选定的值，与 batch 规则中的抽样范围不同。

只准备第 0、5 帧的标签任务：

```bash
"$GEN_PY" -m human_data_generator.run prepare \
  --scene config/scene_example.json \
  --name city_labels --frame 0 --frame 5
```

只准备一个相机组的单帧图片预览：

```bash
"$GEN_PY" -m human_data_generator.run prepare \
  --scene config/scene_example.json \
  --name city_preview --frame 5 --rig rig_side --preview
```

预览准备结果在 `outputs/previews/city_preview/`。要得到图片，再用 `run-prepared --root outputs/previews/city_preview` 并指定 GPU 和预算。预览用于看图；需要训练标签时使用普通任务或 `--frame` 选帧任务。

## 5. 使用交互式编辑器

启动服务：

```bash
"$GEN_PY" -m human_data_generator.editor.server --host 127.0.0.1 --port 8766
```

在运行服务的机器上打开 <http://127.0.0.1:8766>。如果浏览器在另一台机器上，需要先把远程 8766 端口转发到本机。

编辑器中可以选择四类组件，修改人物路线、相机组、帧率和帧数：

- **CPU 检查**：检查配置，不渲染。
- **保存修订**：保存当前场景。
- **预览当前帧**：使用 GPU 渲染选定相机组的一帧。
- **生成所设帧段**：使用 GPU 生成当前场景的数据。

切换环境会保留人物组件，但需要重新检查路线和机位是否适合新环境。地图中的绿色区域表示道路范围，不保证其中所有障碍物都已检查。

修改后端代码后，需要重启服务才能加载新版本；不要在任务运行中重启。

## 6. 生成结果在哪里

| 目录 | 内容 |
|---|---|
| `outputs/scenes/<批次名>/` | 自动抽样得到的场景 JSON |
| `outputs/cache/components/` | 按需构建、可复用的组件几何缓存 |
| `outputs/runs/<任务名>/` | 正常任务的准备文件、渲染结果和标签 |
| `outputs/previews/<预览名>/` | 单帧图片预览任务 |
| `outputs/acceptance/` | 开发验证记录，不是日常生成数据的入口 |

### 准备完成时

单个相机组的任务有六个文件：

```text
outputs/runs/walk_001/
├── scene.json                 # 最终选定的场景
├── recipe.json                # 解析后的人物与时间信息
├── scene_matrices.npy         # 逐帧人物变换矩阵
├── run_config.json            # 选定帧、相机组及执行配置
└── rigs/
    └── rig_side/
        ├── rig.json           # 两个相机的标定与位置
        └── planned_regions.json  # 各帧的裁剪范围
```

增加相机组时，只增加对应组的相机和裁剪文件，共享的人物数据不重复存储。

未启动的准备目录可以整体移动，再用 `run-prepared --root` 指向新位置。外部场景、组件缓存、标定、模板等仍需可访问；准备目录不是包含所有原始资产的独立压缩包。

### 渲染完成时

先看任务根目录的 `data_complete.json` 和 `observations.json`。前者记录数据完成情况，后者是训练数据清单。

具体图片和标签在 `rigs/<相机组>/frame_<帧号>/<相机>/` 下，常用文件为：

| 文件 | 内容 |
|---|---|
| `roi.png` | 实际渲染的 RGB 区域 |
| `person_<人物ID>_mask.png` | 对应人物的二值 mask |
| `mamma_512.npz` | 512 点坐标、可见性及必要的有效性信息 |
| `annotations.npz` | 人物、帧和时间等配套信息 |
| `raster_storage.json` | 图片区域与完整全景图之间的位置关系 |

这里的 ROI 是从完整全景图中裁出的区域；**512 点坐标仍以完整 ERP 全景图为基准**，不能直接当成 `roi.png` 内的坐标。

正常生成会自动导出清单。如果当前格式任务的数据已齐全、只是尚未导出清单，可以执行：

```bash
"$GEN_PY" -m human_data_generator.run export --root outputs/runs/walk_001
```

该命令不重新渲染，也不复制图片；已有 `observations.json` 时会拒绝覆盖。

### 交给 MAMMA 读取

把 [MAMMA reader 配置](../mamma_retrain/configs/retrain/mamma_saved_gt_reader.json) 中的 `manifest` 指向这次任务的 `observations.json`。

训练端只需要清单及其引用的图片、mask 和标签，不需要原始人物、动作、衣物或发型缓存。训练操作见 [MAMMA 当前用法](../mamma_retrain/docs/MAMMA_CURRENT_ENTRY.md)。

## 7. 常见情况与当前验证范围

| 遇到的情况 | 如何处理 |
|---|---|
| `No module named human_data_generator` | 在项目目录重新执行第一节的 `export PYTHONPATH` |
| 准备后没有图片 | `prepare` 只准备几何；图片由渲染步骤生成 |
| 任务名或目录已存在 | 换一个新任务名；已启动任务不支持直接重复执行 |
| 动作范围越界 | 减小动作起点或输出时长，或选择更长的动作 |
| 组件不匹配 | 检查候选 ID、动作类别和锁定衣物的限制 |
| 旧配置被拒绝 | 按当前示例重新配置；不兼容旧 `asset`、`motion_behavior` 或旧准备目录格式 |

当前已有 CPU 几何、配置、数据导出和模拟调用验证；**新组件和整理后的链路尚未完成真实 Blender 渲染验收**。新衣物、发型的最终观感仍需看实际图片。可信遮挡负例也仍待验证，不能仅凭流程检查通过认定数据已经适合正式训练。

实现细节和已有验证记录放在以下文档，日常使用无需先读：

- [四类组件接入](docs/COMPONENT_SELECTION_A1_20261023.md)
- [准备目录整理](docs/PREPARED_LAYOUT_SIMPLIFICATION_A1_20261023.md)
- [生成与读取流程](docs/PIPELINE_SIMPLIFICATION_A1_20261023.md)
