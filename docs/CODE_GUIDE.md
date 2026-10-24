# 用一个例子读懂生成器

这份文档讲的是**数据怎样从场景配置走到训练清单**，以及需要修改行为时该从哪里读代码。想直接操作生成器，请看 [README 使用手册](../README.md)。

先看现成的 [scene_example.json](../config/scene_example.json)：它指定了城市环境、1 个人（`person_id: 1`）、身体 `M01`、动作 `APD6526`、衣物 `M01_simple`、发型 `M01_groom`、行走路线，以及 `rig_side` 这一组相机。场景共有 16 帧，每秒 8 帧。假设我们只选择第 0 帧和第 5 帧来生成标签，下面就是它会走过的流程。这里是**流程示例**，不表示本文执行了渲染。

## 第一件事：决定“拍什么”

场景 JSON 像一张拍摄单：谁出现、穿什么、做什么动作、走哪条路、相机放哪里，都已经写定。入口是 [`run.py`](../src/human_data_generator/run.py)。

`plan` 只回答“这张拍摄单能不能执行”：检查人物和资源是否存在、动作时间是否越界、路线和相机参数是否合理。它不会生成图片，也不会构建人物几何。检查场景的代码主要在 [`runtime/scene.py`](../src/human_data_generator/runtime/scene.py)；检查身体、动作、衣物、发型能否组合的代码在 [`components.py`](../src/human_data_generator/components.py)。

如果不想逐份手写拍摄单，可以让 [`batch.py`](../src/human_data_generator/batch.py) 按 [batch_rules.json](../config/batch_rules.json) 抽样。它的产物仍是场景 JSON。**批量抽样到这里就结束了**；后面的步骤使用选定的组件，不会重新抽一次。

编辑器做的是同一件事的可视化操作：页面在 [`editor/index.html`](../src/human_data_generator/editor/index.html)，表单和状态在 [`editor/editor.js`](../src/human_data_generator/editor/editor.js)，地图拖动在 [`editor/editor_map.js`](../src/human_data_generator/editor/editor_map.js)，HTTP 接口在 [`editor/server.py`](../src/human_data_generator/editor/server.py)。页面设置的随机规则交给同一个 `batch.py` 试抽和导出，没有第二套浏览器采样算法。

## 第二件事：把拍摄单变成可执行计划

`prepare` 会按选中的组件准备人物几何，并算出人物在各帧的位置、相机参数和需要渲染的局部范围。人物组合的实现主要在 [`composition.py`](../src/human_data_generator/composition.py)，逐帧和相机计划在 [`runtime/multirig_prepare.py`](../src/human_data_generator/runtime/multirig_prepare.py)。这一步可能在 CPU 上构建可复用的组件缓存，**仍没有拍出 RGB 图片**。

如果任务名叫 `city_labels`，准备后的任务目录是：

```text
outputs/runs/city_labels/
├── scene.json                 原拍摄单的快照
├── recipe.json                已解析的人物和动作时间
├── scene_matrices.npy         人物逐帧的位置与姿态数值
├── run_config.json            本次选第 0、5 帧及相机组等执行选项
└── rigs/rig_side/
    ├── rig.json               这一组相机的位置与参数
    └── planned_regions.json   第 0、5 帧计划拍摄的局部范围
```

此时没有 `roi.png` 很正常；`planned_regions.json` 只是拍摄范围的**计划**，不是图像。

## 第三件事：真正拍摄、整理图片和标签

执行 `render` 或 `run-prepared` 时，[`run.py`](../src/human_data_generator/run.py) 才启动渲染任务。Blender 根据计划装配场景；第 0 帧更新一次人物，再让 `cam2` 和 `cam4` 各拍一次；第 5 帧也是如此。**两帧 × 两台相机 = 四份相机结果**。负责装配和相机循环的代码分别在 [`render/assembly.py`](../src/human_data_generator/render/assembly.py) 和 [`render/multirig.py`](../src/human_data_generator/render/multirig.py)。

渲染器先产生临时缓冲。随后 CPU 端的 [`consumer.py`](../src/human_data_generator/consumer.py) 把它们整理成每台相机可用的文件。例如第 0 帧 `cam4` 的目录可能是：

```text
rigs/rig_side/frame_000/cam4/
├── roi.png               这台相机拍到的局部 RGB 图
├── person_1_mask.png     1 号人物的黑白区域图
├── mamma_512.npz         有序 512 点坐标、可见性及检查信息
├── annotations.npz       人物、相机和时间信息
└── raster_storage.json   局部图在完整全景中的位置
```

同一帧、同一相机拍到两个人时，RGB 和标签文件由这次相机结果共享，每个人各有自己的 `person_<编号>_mask.png`。某人的 mask 也可能为空；这不意味着删掉这个人，最后仍会记录一次 `absent`（未观察到）。

所有选定帧和相机完成后，[`data/mamma_export.py`](../src/human_data_generator/data/mamma_export.py) 检查文件、人物编号、帧号和相机是否对得上，写出 `observations.json`。在上述单人例子中，它有 **2 帧 × 2 相机 × 1 人 = 4 条观察记录**；若改为两个人，就是 8 条。记录中引用 RGB、该人的 mask 和共享的 512 点标签。MAMMA 读取这份数据包时，不需要再去找原身体或动作资产。

## 两个容易误解的地方

**局部图片与点坐标不是同一张坐标纸。**`roi.png` 只存局部，节省渲染和存储；512 点的 XY 仍标在完整的 5760×2880 全景坐标上。`raster_storage.json` 说明局部图在完整全景中的位置。训练时人物裁剪框从该人物 mask 得到，不需要另做框标注。

**场景帧与动作原始帧不是同一个编号。**例如“拍场景第 5 帧”不保证使用“动作文件第 5 帧”。`recipe.json` 明确记下每个人实际取用的动作时间。多个相机拍同一场景帧时，共用同一个人物状态。

## 想改东西，从哪里开始

不用一次读完所有源码。先读 [`run.py`](../src/human_data_generator/run.py) 看任务顺序；遇到具体问题，再查对应文件：

| 你想弄明白或修改的事 | 先看这里 |
|---|---|
| 为什么抽到某个身体、动作、衣物或发型 | [batch_rules.json](../config/batch_rules.json)、[`batch.py`](../src/human_data_generator/batch.py) |
| 为什么某个已选组合不能用 | [`components.py`](../src/human_data_generator/components.py) |
| 人物怎样变形、穿衣和戴发型 | [`composition.py`](../src/human_data_generator/composition.py) |
| 人物何时走到哪里、相机拍哪里 | [`runtime/scene.py`](../src/human_data_generator/runtime/scene.py)、[`runtime/multirig_prepare.py`](../src/human_data_generator/runtime/multirig_prepare.py) |
| 为什么图片或 mask 不对 | [`render/multirig.py`](../src/human_data_generator/render/multirig.py)、[`consumer.py`](../src/human_data_generator/consumer.py) |
| 为什么 512 点或观察清单不对 | [`data/mamma_labels.py`](../src/human_data_generator/data/mamma_labels.py)、[`data/mamma_export.py`](../src/human_data_generator/data/mamma_export.py) |
| 为什么任务显示失败或未完成 | [`run.py`](../src/human_data_generator/run.py) 和任务目录中的 `status.json`、日志 |

这里的 **mask** 是“哪些像素属于这个人”的黑白图；**ROI** 是相机实际保存的局部画面；**rig** 是一组一起移动的相机。记住这三个词，就足以顺着上面的例子继续读代码。
