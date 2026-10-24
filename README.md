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
GEN_PY=/home/yy/conda_envs/densekp_render/bin/python
```

每次新开终端，先执行上面这三行。生成时使用的 Python 也会成为 CPU consumer 的解释器，必须能读取 OpenEXR；本机 `densekp_render` 已通过真实渲染验证，`camerahmr` 缺少 `OpenEXR`。换到其他机器时，需要调整 Python 路径和 [resources.json](config/resources.json) 中的资源位置。

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
  --name key_batch_001 \
  --count 10
```

`--count 10` 表示**每个基准场景**本次抽 10 个变体；改这个数字即可决定本次规模，不需要重写规则文件。结果放在 `outputs/scenes/key_batch_001/`。每份场景 JSON 都已经确定了人物、动作、衣物、发型、路线和机位。命令会打印生成文件的路径。

**这一步只写 JSON，不构建几何、不渲染。**当前批量入口也不会自动执行后面的准备和渲染。

默认规则有城市单人步行、城市单人骑行、城市双人相向经过、河岸单人骑行。双人规则当前两人都使用骑行动作。

### 第二步：检查一份场景

以默认规则生成的第一份场景为例：

```bash
SCENE=outputs/scenes/key_batch_001/batch_city_single_walk_v0001.json

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

## 3. 怎样设置随机规则

推荐在编辑器里先布置并保存基准场景，再到“随机变化”页设置候选和范围，试抽后导出规则。导出的单份 JSON 可以直接传给 `human_data_generator.batch --rules`；重新导入编辑器时，基准场景和随机设置都会恢复。

城市规则做完后要单独制作河岸规则，可在河岸场景中点击“从当前场景开始新规则”。它只替换页面中的规则草稿，不会修改已导出的城市文件；新规则默认全部固定，再按需打开随机项。

[batch_rules.json](config/batch_rules.json) 是可直接运行的例子，格式为 `human_batch_v4`。根部的 `seed` 是整批随机种子，`templates` 是基准场景列表。规则只描述如何变化，生成数量由运行批量命令时的 `--count` 决定。每项有：

- `template_id`：变体命名前缀。
- `base_scene`：完整的 `human_scene_v2` 场景快照，包含全部路线控制点、人物和相机组。移动规则文件后，基准场景仍在文件里。
- `route_offset_xy_m`：所有人的路线一起平移，格式为 `[[X 最小, X 最大], [Y 最小, Y 最大]]`，单位米；不改变控制点之间的形状和人物相对位置。
- `people`：只列需要随机变化的人物。每项以 `person_id` 指向基准人物；`route_offset_xy_m` 是这个人物额外的路线平移，可能改变人物间距；`speed_m_s`、`motion_start`、`start_progress_m` 分别是速度、源动作起始帧、沿路线已前进的距离的**绝对取值范围**。
- `rigs`：只列需要变化的相机组。`offset_xy_m` 和 `offset_yaw_deg` 是相对于基准位置/朝向的偏移范围；组内 cam2/cam4 的标定关系不变。

**未写的变化项保持基准值。**例如只设置某个人的 `"speed_m_s": [0.9, 1.2]`，他的身体、路线、动作和其他人物都不会因此随机变化。范围上下界相同可以在明确开启随机设置后固定该值；未开启时无需写 `min=max`。

人物的 `components` 可按身体、动作、衣物、发型分别设置候选。仅列要随机选的类型；未列的类型保持基准组件。候选中的 `ids: []` 表示使用满足筛选条件的全部可用资源，非空 ID 列表表示限定候选。身体可用 `default_versions_only` 限定默认版本；动作 `behavior` 选 `walk` 或 `ride`；衣物 `kinds` 选可重绑的 `simple_barycentric` 或有真实身体/动作/源帧限制的 `native_locked`。错误候选和冲突会报错，不会自动换衣物或偷偷跳过场景。

相同规则文件、资源、seed 和变体序号会抽出相同场景；把 `--count` 从 2 改成 10，前两个变体不变。候选 ID 列表的顺序与模板列表的顺序不改变各模板自己的抽样。CLI 只写 `human_scene_v2` 场景 JSON，之后仍需按第二节检查、准备和渲染。旧 `human_batch_v3` 规则含固定 `count`，当前入口会明确拒绝；已保存的旧规则可一次性去掉各模板的 `count` 并把 `schema` 设为 `human_batch_v4`。更早的规则需按当前示例重新制作。

查看可用资源 ID 和限制：

```bash
"$GEN_PY" -m human_data_generator.components
```

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

在运行服务的机器上打开 <http://127.0.0.1:8766>。若浏览器在另一台机器上，先将远程端口转发到本机。端口占用时，把命令和浏览器地址中的 `8766` 一起改成空闲端口。已有服务可能读到新版页面，但不会自动加载新版 Python 后端；若相机箭头出现、时间轴旁却提示“旧后端”，请在空闲端口另启新版服务并打开那个地址。

1. 在“基准场景”页选择身体、动作、衣物和发型；编辑多控制点路线和相机组。点击“CPU 检查场景”后，可“保存场景修订”。已有修订可从下拉列表载入，也可导入场景 JSON。单帧 GPU 预览和整段 GPU 生成仍在右侧的“真实预览与生成”折叠区，只有显式点击才会启动真实任务。
2. 到“随机变化”页，打开想变化的组件、路线、速度、动作起点或相机组。默认全部固定。可添加多个基准场景；一份规则文件会把它们一起保存，不限制以后抽样数量。
3. 点击“试抽配置并检查”，每个基准只抽前两个变体做 CPU 场景检查。点击结果可在地图查看路线与相机位置，不会准备几何或渲染。检查通过后点击“导出规则文件”；服务器会将规则保存到 `outputs/rules/<导出名称>.json`，名称可用英文小写、数字、下划线或连字符。页面同时提供下载链接和批量命令。运行命令时修改 `--count`，决定每个基准本次生成多少个。导出的 JSON 可重新载入继续修改。

[查看编辑器界面示例](docs/images/editor_rules_20261024.png)。

地图绿色区域表示道路候选范围，不保证障碍物已全部检查。切换环境时人物组件、路线和机位坐标都先保留；城市路线可能因此画在河岸地图外。此时点“把人物和相机整体移到当前环境附近”，可保留路线形状及相对位置并整体平移，再检查道路和机位。新增人物会从当前环境的默认路线附近出现。改变场景后，旧预览图片会被清除，避免误认成当前设置。
CPU 检查允许不同人物重叠，也允许人物路线经过相机组位置；仍会检查路线是否在当前道路候选区域，以及动作和资源是否可用。
拖动右侧“当前帧”滑块，地图上的橙色圆点会沿路线移动，显示各人物这一帧的计划位置和方向；浅色圆点仍是可拖动的路线控制点。切换帧会清除上一帧的图片。地图位置由场景路线、速度和帧率计算，人物姿态与真实画面需要点击“GPU 预览当前帧”查看。
相机组旁的青色 `cam2` 和紫色 `cam4` 箭头分别表示两台鱼眼相机的物理光轴方向，数字是该相机组的 yaw 角度。箭头用于看朝向，不按相机间距比例绘制；双机的相对标定保持不变。

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

先看任务根目录的 `status.json` 和 `observations.json`。`status.json` 的 `state: "data_complete"` 表示图片、标签和清单已经完成；`observations.json` 是训练数据清单。渲染过程事件追加在 `events.jsonl`，错误详情可查看 `blender.log` 和 `consumer_worker.log`。可选视频失败时，数据仍可完成，视频结果记录在 `status.json` 的 `video_status` 中。

具体图片和标签在 `rigs/<相机组>/frame_<帧号>/<相机>/` 下，常用文件为：

| 文件 | 内容 |
|---|---|
| `roi.png` | 实际渲染的 RGB 区域 |
| `person_<人物ID>_mask.png` | 对应人物的二值 mask |
| `mamma_512.npz` | 512 点坐标、可见性及必要的有效性信息 |
| `annotations.npz` | 人物、帧和时间等配套信息 |
| `raster_storage.json` | 图片区域与完整全景图之间的位置关系 |

这里的 ROI 是从完整全景图中裁出的区域；**512 点坐标仍以完整 ERP 全景图为基准**，不能直接当成 `roi.png` 内的坐标。

ERP 的最左与最右实际上相连。人物跨过这条接缝时，对应相机会保存一条全宽图片，再按该人物的 mask 做连续的周期裁剪；仍是一人一条观察，不拆成左右两人。普通机位继续保存较小的 ROI。全宽条带文件较大，打开原图时人物会分处两端，训练 reader 会按清单中的裁剪规则把它接起来。

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

已有一次双人、两帧、一个相机组的[真实渲染验证](docs/FIRST_CURRENT_RENDER_A1_20261023.md)：图片、标签、导出及 MAMMA 读取均已走通。后续源码与运行记录整理经过 CPU 和模拟调用检查，**尚未用新代码再次真实渲染**。这次小样本也不能代表所有组件组合的画面质量；可信遮挡负例仍待验证。

想读源码，可从[源码阅读指南](docs/CODE_GUIDE.md)开始。其他实现细节和已有验证记录放在以下文档，日常使用无需先读：

- [四类组件接入](docs/COMPONENT_SELECTION_A1_20261023.md)
- [准备目录整理](docs/PREPARED_LAYOUT_SIMPLIFICATION_A1_20261023.md)
- [生成与读取流程](docs/PIPELINE_SIMPLIFICATION_A1_20261023.md)
