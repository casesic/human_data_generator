# 准备目录整理（2026-10-23）

当前 `human_scene_v2` 的单 rig 准备结果只写六个文件：根目录的 `scene.json`、`recipe.json`、`scene_matrices.npy`、`run_config.json`，以及 `rigs/<rig_id>/rig.json` 和 `planned_regions.json`。多 rig 共享人物、时间和矩阵；每组只新增自己的相机参数与 ROI 计划。新任务不写 `job.json`、`multirig.json`、`camera_configs/`，旧准备结构会明确拒绝，不执行隐式转换。

`run_config.json` 唯一保存有序 `selected_frames`、`rig_ids`、执行选项和启动时建立的 `budget_seconds`、`deadline_monotonic`。装配入口接收任务根目录，直接读取固定的 recipe、矩阵及每组 rig 文件；相机渲染函数直接接收该 rig 中的 cam2/cam4 对象。旧的任务重处理字段和未使用的重处理分支一并删除；已启动任务仍拒绝重复运行。`source_manifest` 中的 recipe 与源缓冲引用、缓冲归属标记和导出清单的包内路径均以任务根目录为基准。外部组件缓存、环境 `.blend`、标定、512 模板与可执行文件仍按资源配置指向原位置，不属于可搬移的数据包。单纯日志中的绝对路径仅用于诊断，不作为恢复或定位依据。

已用 CPU 夹具及 mock 验证：单 rig 六文件、双 rig 八文件、预览和非连续选帧；实际搬移未启动任务并删除旧路径后，从另一工作目录加载并进入 mock 启动点，预算在启动时写入；装配逐帧更新共享人物矩阵，multirig 回调按 rig 顺序传递帧号，ROI 层逐个传递 cam2/cam4 参数对象；合成 RGB/range/instance 的 consumer→标签→`mamma_bundle_v1` 导出及身份/时间/模板拒绝；临时缓冲的正确归属清理、外部或其他任务路径拒绝、标记篡改拒绝和失败现场保留。现有 `tests/test_current_pipeline.py` 共 14 项通过（最终完整运行 7.1 秒；本批多轮 CPU 测试累计约一分钟）。Python 编译、JSON 解析与 diff 检查也通过。测试未启动 Blender、GPU、模型、训练、编辑器服务或新 RGB 渲染；真实渲染与消费子进程集成仍待单独验收。

当前 CLI 示例：`python -m human_data_generator.run prepare --scene config/scene_example.json --name city_check`；未来真实执行用 `python -m human_data_generator.run run-prepared --root outputs/runs/city_check --gpu-uuid GPU-UUID --budget-seconds 1800`，本批未运行。已有当前格式数据可用 `python -m human_data_generator.run export --root outputs/runs/<任务名>` 导出。MAMMA 数据包合同与 reader 未改；旧资产和历史输出未移动或覆盖。`city.json` 及装配中预先存在的 `expected_blend_stat` 删除保持原样，本批未恢复该校验。
