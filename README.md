# human_data_generator

这是人体合成数据生成项目。当前场景入口是一个 `human_scene_v2` JSON。每个人物实例分别选择 `body`、`motion`、`clothing`、`hair`，再指定动作起始帧与路线；场景另指定环境、相机组、帧率、帧数和 seed。设备与时间预算只在执行时指定。生成端不导入 DenseKP 或 MAMMA 训练代码。旧 `asset` 字段和旧场景格式明确拒绝。

正常输出是双机 ERP ROI RGB、每个人的二值 mask、有序 512 点完整 ERP XY 和 visibility（1 可见、0 被遮挡、-1 未知）。框由目标 mask 按 `mask_bbox_inclusive_v1` 派生。每个 `episode/rig/frame/camera/person` 都有观察槽，空 mask 记为 absent。坐标有效性由几何、物理相机域、ROI 与实际 crop 共同决定；visibility 为 -1 不自动取消有效坐标监督。接触标签不生成。当前 visibility 负例尚未资格确认，结构整理不改变该事实。

[config/resources.json](config/resources.json) 定位四份[组件目录](config/catalog/)以及场景、标定、模板和 Blender。目录是资源定位和技术能力说明，不是人工资格白名单；准备时检查目标形状、标准 SMPL-X 面片顺序、时间范围、衣物绑定、头发与模板。旧资产保持只读，按需制作的短窗口缓存写 `outputs/cache/components/`，新任务写 `outputs/runs/<任务名>/`，预览写 `outputs/previews/<预览名>/`。用户场景例子见 [config/scene_example.json](config/scene_example.json)，规则批量例子见 [config/batch_rules.json](config/batch_rules.json)。

在本项目目录使用已有带 numpy、Pillow、OpenCV、joblib 的 Python；本机可用 `/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python`。无需安装本包：

```bash
cd /mnt/data1/yy/home/yy/Projects/human_data_generator
export PYTHONPATH="$PWD/src"
GEN_PY=/mnt/data1/yy/home/yy/Repos/mamma/.venv/bin/python

# 只读查看四类组件、数量、可用状态、动作帧范围和限制
$GEN_PY -m human_data_generator.components
$GEN_PY -m human_data_generator.components --kind motions

# 不启动 Blender/GPU；plan 只检查场景与源动作范围
$GEN_PY -m human_data_generator.run plan --scene config/scene_example.json

# CPU 按四类选择构建短窗口几何并规划 ROI；只创建一个新任务目录
$GEN_PY -m human_data_generator.run prepare --scene config/scene_example.json --name city_check

# 若只需某帧的标签数据，可多次给 --frame；预览单帧需同时选一个 rig
$GEN_PY -m human_data_generator.run prepare --scene config/scene_example.json --name city_labels --frame 0 --frame 5
$GEN_PY -m human_data_generator.run prepare --scene config/scene_example.json --name city_preview --frame 5 --rig rig_side --preview

# 将来明确授权真实生成时，二选一；本轮没有执行
$GEN_PY -m human_data_generator.run render --scene config/scene_example.json --name city_run --gpu-uuid GPU-UUID --budget-seconds 1800
$GEN_PY -m human_data_generator.run run-prepared --name city_check --gpu-uuid GPU-UUID --budget-seconds 1800
```

`prepare` 不写预算截止时钟；`render` 和 `run-prepared` 在实际启动时设置截止时间，并拒绝已启动或已有结果的任务。生成成功后先检查相机产物并写 `observations.json`、`data_complete.json`。可选 `--video` 在数据完成后制作展示视频；视频失败保留数据包。已有但尚未导出的当前格式任务可执行 `$GEN_PY -m human_data_generator.run export --name <任务名>`，不复制 RGB/NPZ。旧 job/recipe/manifest 不由新入口解析。

小批量规则采样只生成场景配置，不渲染：

```bash
$GEN_PY -m human_data_generator.batch --rules config/batch_rules.json --name key_batch_001
```

输出中的每份场景 JSON 可再交给上述 `plan/prepare/render`。批量规则从可用身体、对应走路/骑行动作、普通绑定衣物和发型目录独立采样；锁定衣物不会被随机误配。同一规则、资源、总 seed 与场景编号得到同样配置；四类组件、轨迹、相机和渲染种子按用途派生。

编辑器将来也从本项目独立启动：

```bash
$GEN_PY -m human_data_generator.editor.server --host 127.0.0.1 --port 8766
```

现有编辑器进程不会自动加载新代码；本轮没有启动或重启服务。新后端的四类选项保存到同一场景格式，换环境不会重置组件；`check` 只读，只有点击预览或生成才会请求 GPU。地图绿色区域是道路候选，不表示未知障碍已完成净空核验。

A1 在 `mamma_retrain/configs/retrain/mamma_saved_gt_reader.json` 指向交付包的 `observations.json`；`MammaSavedGTDataset` 只需要该清单及其相对引用的 RGB、mask、`mamma_512.npz`、`annotations.npz`，无需生成端缓存在线。具体字段与 CPU 验证范围见 [简化报告](docs/PIPELINE_SIMPLIFICATION_A1_20261023.md)。

四类选择的资源数量、已有短窗口 CPU 组合和客观限制见 [组件接入报告](docs/COMPONENT_SELECTION_A1_20261023.md)。普通简单衣物可按目标身体重绑，静态发型可按目标头皮转移。M03/KIT11 的 BEDLAM 衣物是身体、动作、源时间锁定的已有缓存，只能在该范围内使用；头发仍可独立选择。目录中的原始动作线索与旧通勤衣物/发型缺件版本会显示原因，不能作为可直接生产的选项。真实 Blender 渲染尚未用新组件做过验收。
