# A3 生成器独立化（2026-10-22）

> 2026-10-23 更新：本报告保留迁移当天的历史状态。下文关于资产资格注册表、未登记缓存拒绝和 A1 注册依赖的规定已取消；当前所有兼容 SMPL-X 人物可自由组合，执行自动格式/模板检查即可。当前用法见 README；CPU 验证见 `outputs/acceptance/asset_registry_removal_20261023/checks.json`。

`human_data_generator` 是新的生成端入口。编辑器、配方冻结、相机/身体几何、ROI 规划、Blender 脚本、CPU consumer、512 标签和 observation 导出均从自己的 `src/` 运行。`densekp_retrain` 保留训练与历史实验；上一批新增的 `scene_geometry`、`mamma_export` 旧导入点现为指向新项目的薄包装。旧编辑器/实验命令暂作为历史路径保留，避免影响未停止的旧服务；新项目不反向调用它们。没有移动、重写旧 recipe、RGB/NPZ、资产或失败现场。

| 旧路径 | 新路径/状态 |
| --- | --- |
| `densekp_retrain/tools/scene_editor/{server,cli,core,index.html}` | `src/human_data_generator/editor/`，新任务写 `outputs/editor/` |
| `tools/experiments/run_multirig_roi_a1.py` 与 `src/densekp_retrain/runtime/multirig_prepare.py` | `src/human_data_generator/run.py` 与 `runtime/multirig_prepare.py` |
| 旧 Blender 主/常驻预览与装配脚本 | `src/human_data_generator/render/`、`rendering/blender/` |
| ROI consumer、存储、512 标签、manifest 导出 | `consumer.py`、`data/`、`imaging/`、`export.py` |
| 上游 `densekp.core.datasets.image_layout` 的生成侧布局合同 | 独立最小 `data/image_layout.py`；不加载 DenseKP Dataset/loss |
| 旧七框 probe、DenseKP R138 实验 | 留在旧仓库，仅历史兼容；正常 MAMMA 不要求 R138 或七框表 |
| 旧 `src/densekp_retrain/data/{scene_geometry,mamma_export}.py` | 薄兼容包装，调用新项目同名生成侧实现 |

资源与代码定位分开。`config/resources.json` 明示旧 prepared 场景、blend 所属资源根、MAMMA 有序模板、A1 已审核资格注册及 Blender 可执行文件；它们是只读数据/工具引用。新项目源代码没有对 `densekp_retrain`、`densekp`、`mamma_retrain` 或旧 `tools` 的 import，也不动态载入旧 Python。Blender 两个入口通过新项目 `src` 自定位；消费者的 `PYTHONPATH` 显式指向新项目。旧 blend 若含 Python 驱动/脚本，其运行行为需在未来获准的 Blender 验收中确认，不能把本次 CPU 检查写成已验收。

旧 `scene_editor_recipe_v1` 由新编辑器直接解析，冻结时保留人物身份、整数时间/来源、变换与 rig 相对外参。已核对旧城市 r3 的 frame 15、rig_back：人物矩阵差值 0，cam2/cam4 预留 ROI 及 rig JSON 一致。新常规准备不计算 world_densekp 或 expected_in_view；消费者仍写 A1 所需的轻量 `annotations.npz`，包括人物顺序、相机矩阵、frame/time/source/episode 信息。单帧预览走 `preview_rgb_v1`，未知 512 资产不被伪装成合格；完整 MAMMA 走 `mamma_512_v1`，未注册缓存拒绝。旧 recipe 中遗留 DenseKP 字段可读取，但不作为新消费的强制输入。

正常产物保留旧 `mamma_observation_v1`：逐 episode/rig/frame/camera/person 一行，RGB 与目标 mask 引用、`mamma_512.npz` 中的有序 XY/visibility、physical/ROI/crop 支持、身份整数时间与来源。观察 ID 根据实际键构成；absent 行保留原因。新 `mask_native_v2` 不制作 `mamma_box_support.json`，导出仍核查实际 mask 与完整滤波源支持；旧 `seven_box_probe_v1` 有表时继续读。mask 框按 `mask_bbox_inclusive_v1`；旧 exclusive 框的一像素差异没有追平或重写。contact/floor-contact 不属于新训练目标，旧审计字段暂兼容保留。512 模板资格只由审核注册给出，不由 person_id、性别或顶点数推断。

A1 对接使用其现有 `MammaSavedGTDataset`、`mamma_manifest_reader_v3` 配置与新清单路径，无需改 A1 reader。新项目只读重导出 e03 六份旧标签为 [observations.json](../outputs/acceptance/migration_a3_20261022/observations.json)：12 槽、10 observed、2 absent，两人均为 target，身份/时间、图像/mask/NPZ 引用与旧清单一致。A1 已从新清单初始化并实际读取 1 条，返回有序 512 目标及可见性。合成 fixture 的人物 7/9 与两 rig 只证明代码遍历和拒绝行为，注册明确写 `fixture_only`，不能作为真实资产资格。

验收见 [acceptance.json](../outputs/acceptance/migration_a3_20261022/acceptance.json)。CPU 数值/脚本累计低于 5 分钟，低于 30 分钟上限；实际 A1 图像读取 1 次。旧导出 fixture 与人物变换包装最小回归通过；所用现有 Python 没有 pytest，故直接调用原测试函数，未安装依赖。Blender/GPU/模型/训练/新 RGB 渲染均 0。隔离 socket 受本机沙箱限制，编辑器 `/` 与 `/api/default` 使用无监听 Handler 测试；静态页面和默认 recipe 正常。真实独立渲染、完整 31 帧 finish/video、blend 内驱动和 visibility 负例资格仍待后续单独验收。当前 1137 正例、0 可信负例、5007 unknown 没有改变；2194 个自遮挡候选未转为负标签。迁移不意味着训练效果提升。

给总协调与 A1：继续用 `mamma_observation_v1` 与已审核资产注册；新清单内旧路径是只读来源，不能把输出目录或 seed 当独立来源。若 A1 的 reader 合同继续变化，应先核对具体字段再版本化，不应默默改变旧数值。已有旧编辑器进程不会加载本项目；需用户将来另启新端口，当前未代为重启。
