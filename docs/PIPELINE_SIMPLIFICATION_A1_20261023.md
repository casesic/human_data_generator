# 生成与读取主流程简化（2026-10-23）

本轮将正常生产收敛到一套 `human_scene_v1 → human_run_v1 → mamma_bundle_v1` 路径。旧 job、recipe、`mamma_observation_v1` 和七框诊断仍作为历史文件保留，但新入口明确拒绝旧格式，不做运行时转换。

用户编辑的场景只含环境、人物实例与动作资产/起始帧、路线、机位组、帧率、帧数、seed。资产、标定、模板由 `config/resources.json` 定位；旧场景设置已一次性提取为本项目的 `config/environments/*.json`，运行不再解析旧场景配方。资源描述不是资格白名单，资产不再绑定城市或河岸；道路对象等检查由选定环境提供。设备与预算只在 `render` 或 `run-prepared` 指定，GPU 与截止时钟在实际启动时写入一份根目录 `run_config.json`，冻结的场景配方不再被执行阶段改写。一个任务只保存一份人物时间记录和一份 `scene_matrices.npy`；每个 rig 仅保存机位、ROI 计划及结果。准备不会创建根目录帧占位；真正运行时各 rig 创建带归属标记的临时缓冲，成功消费后清理，失败保留。

`human_data_generator.run` 提供 `plan / prepare / render / run-prepared / export`。正常新任务先准备、再做 GPU 资源检查与启动，因此自身日志不会触发“输出目录非空”。`job.json` 只记录选定帧和共享配方，不存会过期的预算副本；三机位一小时等实验特判已删除。预览、少量选定帧标签和完整任务共用准备/执行实现，完整范围取自当前 `frame_count`，时间由 `fps` 和原动作整数时间映射；超过动作缓存直接拒绝。小批量 `human_data_generator.batch` 按总 seed、场景编号和用途稳定采样走路、骑行、双人错位交错及机位扰动，只写确定的场景配置。

消费端只保存 RGB、逐人 mask、轻量 `annotations.npz` 和 `mamma_512.npz`；不写接触占位数组、七框表或独立框增强。数据完成校验与 `observations.json` 导出先于可选视频。新清单保留全部 target 与 absent 原因；顶层保存一次模板点序，逐观察行使用包内相对路径引用共享的帧/相机 RGB、标签与注释，并以 `person_index` 指向个人标签行。visibility 仅在 NPZ 保存，数值仍为 1/0/-1。A1 的 `MammaSavedGTDataset` 只读取这些包内文件，不再访问身体缓存、面片、旧 timeline 或场景矩阵；原资源路径仅可作为来源 ID 字符串，不作为文件依赖。reader 继续核对模板、身份、整数时间、相机、ROI、数值及 mask 派生 crop 的完整滤波支持。训练只保留坐标/uncertainty 和 visibility loss，接触头为原权重兼容而冻结。

删去了旧 `editor.cli`、常驻预览专用脚本、旧盒增强/布局/观察适配模块及原 one-off 测试；新编辑器服务使用 `index_current.html` 与当前场景格式，并在点击任务时调用同一 `run`。原 `index.html` 静态路径保留原内容，供尚在运行的旧进程继续读取。旧 `resources.json` 中用于执行旧 prepared 目录的字段、忽略式资格参数、`source_dirs`、固定 31 帧/10 Hz 和读回诊断参数不再进入正常调用链。历史 outputs、资产、blend、动作缓存及上游 MAMMA/DenseKP 源码未移动、覆盖或回滚。

有界 CPU 验证见 [checks.json](../outputs/acceptance/pipeline_simplification_a1_20261023/checks.json)：当前示例完整 16 帧/8 Hz 准备通过；走路、城市双人交错、河岸规则场景在身体/衣物道路检查后完整准备通过，交错场景的采样动作起始帧也通过整段准备；双 rig 共用一份人物矩阵；mock 验证新建任务与先准备后执行的启动顺序、预算和 Blender 脚本路径；合成 RGB/range/instance 缓冲经 consumer→导出得到 8 槽、6 observed、2 absent，A1 reader 及正常 `plan/check-data` 命令从无原资产依赖的包读出样本。未知与遮挡 visibility 不取消可靠坐标；错误人物顺序、时间/相机、模板、缺文件和 crop 支持不足均拒绝。纯 CPU loss 算例确认 unknown 时坐标仍有 512 点、visibility 为 0 点监督。数值脚本含失败修正累计约 175 秒，低于 30 分钟上限；真实旧样本图像读取 0 次，合成 Dataset 取样 5 次。

本轮未启动 Blender、GPU、模型、分割器、训练或新 RGB 渲染。真实新项目 Blender 装配、ROI 消费时序与视频仍待一次后续授权的短任务验收；walking 源根轨迹重定位也需在真实渲染中核对。旧小试仍只有 1137 个可见正例、0 个可信负例、5007 个 unknown；这次没有改 visibility 判定或宣称训练效果。A1 默认 reader 清单指向未来 `key_batch_001` 输出，数据生成前正常 plan 会明确因清单不存在而停止。

下一次真实生成时，在本项目目录按 README 先运行 `batch` 产出场景配置，选择其中一份 `run plan` 与 `run prepare`；获批并确定 GPU UUID/预算后对未启动的准备任务执行 `run run-prepared`。本轮未自动开启该步骤，也未重启现有编辑器服务。
