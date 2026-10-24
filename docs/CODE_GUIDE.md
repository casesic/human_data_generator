# 生成器源码阅读指南

先从 `run.py:main` 看 CLI 分发。`plan` 只检查场景和资源；`prepare` 固定组件、人物时间与轨迹，计算共享矩阵和各相机组的 ROI；`render` 是 prepare 后执行，`run-prepared` 执行已有任务；`export` 从现有图片与标签生成观察清单。规则批量场景从 `batch.py:main` 进入，它只抽样并写场景 JSON。编辑器的 HTTP 边界在 `editor/server.py`，它调用相同的场景检查与准备流程。

主调用链如下：

```text
batch.py / editor/server.py / run.py
    → runtime/scene.py                 场景检查、时间和轨迹
    → components.py                    四类资源目录与组合约束
    → composition.py                   按需构建身体、衣物、发型缓存
    → runtime/multirig_prepare.py      共享人物矩阵、相机和 ROI 计划
    → run.py:render_prepared            预算、进程启动、状态与收尾
    → render/multirig.py               每帧一次共享几何更新，各 rig 的 cam2/cam4 渲染
    → render/assembly.py、render/roi.py Blender 装配与 ROI 渲染
    → runtime/frame_consumer.py        Blender 与 CPU consumer 的 JSON 行协议
    → consumer.py                      EXR、RGB/mask、512 标签和文件保存
    → data/mamma_export.py             全员 observations.json
```

`camera/` 保存投影、标定及物理有效域；`imaging/` 保存 ROI、图像布局和临时缓冲归属检查；`data/` 保存人物帧几何、512 点算法、裁剪支持和 MAMMA 清单；`io/` 是 JSON、图像与 EXR 文件边界。`rendering/blender/` 只放 Blender 对象、材质和头发装配，避免普通 CLI 导入 Blender 依赖。

场景帧号是整数，`scene_matrices.npy` 的第一轴对应场景帧；人物的源动作帧由 recipe 中的 timeline 显式指定。共享人物几何在一个场景帧只更新一次，然后依次供选中的 rig 与 cam2/cam4 使用。512 点 XY 使用完整 5760×2880 ERP 坐标；`roi.png` 是局部存储，`raster_storage.json` 记录两者关系。数据包内的相对路径由 `data/mamma_export.py` 写出，MAMMA reader 不需要原始组件文件。

主进程是 `status.json` 的唯一写入者：渲染完成后仍须逐 rig 检查消费结果并成功导出，才标记 `data_complete`。Blender 进程只追加 `events.jsonl`，通过 `runtime/frame_consumer.py` 向常驻 CPU consumer 发送逐帧请求；普通日志不能写入该 JSON 行通信流。帧缓冲的 `.human_data_temporary_source.json` 用于限定可清理范围，失败现场保留。`run_config.json` 保存准备选择和启动时预算，实际截止时间在运行启动时建立。

修改组件候选规则看 `config/batch_rules.json` 与 `batch.py`；修改四类资源发现或组合限制看 `components.py`；修改人物动作、身体或外观构建看 `composition.py`；修改场景时间或轨迹看 `runtime/scene.py`；修改相机/ROI 看 `camera/`、`imaging/roi_planning.py` 与 `render/roi.py`；修改标签或导出看 `data/mamma_labels.py`、`consumer.py`、`data/mamma_export.py`。用户命令和结果文件说明维护在根目录 `README.md`。
