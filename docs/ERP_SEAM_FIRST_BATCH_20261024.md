# 四类城市关键帧：真实生成与 ERP 接缝修复

2026-10-24。按四份已导出的城市规则，各取**第一个变体**，拍开始、交错附近和结束三帧。结果可从[看图页](../outputs/inspection/city_four_first_20261024/inspection.html)打开原图。四个任务均为 `status.json: data_complete`，Blender、真实 CPU consumer、512 点标签和 `mamma_bundle_v1` 清单已完成；没有训练或模型前向。

| 场景 | 帧 | 相机渲染 | observed / absent | visibility 1 / 0 / -1 | GPU | 任务耗时 |
|---|---|---:|---:|---:|---|---:|
| [单人骑车](../outputs/runs/city_four_first_20261024_one_cycle/observations.json) | 0、10、15 | 6 | 4 / 2 | 550 / 0 / 1498 | GPU-d481…61b2 | 114 秒 |
| [双人骑车交错](../outputs/runs/city_four_first_20261024_two_cycle/observations.json) | 0、10、15 | 12 | 18 / 6 | 2725 / 335 / 6156 | GPU-6f04…8cac | 195 秒 |
| [骑车＋走路交叉](../outputs/runs/city_four_first_20261024_cycle_walk_seam_fix2/observations.json) | 0、14、23 | 12 | 14 / 10 | 2100 / 282 / 4786 | GPU-6f04…8cac | 187 秒 |
| [三人骑车](../outputs/runs/city_four_first_20261024_three_cycle/observations.json) | 0、20、23 | 12 | 19 / 17 | 3118 / 0 / 6610 | GPU-d481…61b2 | 274 秒 |

合计 **42 次相机渲染、90 个逐人观察槽，55 observed、35 absent**。absence 均由空目标 mask 记录，没有把跨缝目标改为 absent。visibility 计数只统计 observed 目标的 512 点，不重复计数同相机共享 NPZ。最初骑车＋走路的准备任务因旧接缝限制失败，修复时另有一次 CPU prepare 被旧的 700 万像素 ROI 上限拒绝；两次都在渲染前停止，**额外真实相机调用为 0**。失败准备目录保留，成功任务使用新目录 `cycle_walk_seam_fix2`。

## 接缝为什么报错，怎样修复

原始 frame 0 / rig_side / cam4 的 person 1 身体、衣物和头发确实横跨 ERP 左右边缘，身体全部 10475 个顶点通过物理视野检查。旧代码把 x 的线性跨度接近整幅宽度视为坏几何，准备阶段拒绝；同时它在检查物理视野前已把域外人物加入 ROI，Blender 又独立算了一次 ROI。

现在准备阶段先判定每个人是否可能进入该相机物理有效域：可见顶点直接保留；没有可见顶点时用身体与衣物三角面投影包围盒和有效 mask 作保守交集检查。真正域外人物不再扩大该相机 ROI，但仍有自己的 absent 观察。投影 x 按 ERP 周期求最短覆盖区间；需要越过左右边界或滤波 padding 时，该相机渲染所需纵向范围的**全宽条带**。Blender 直接读取准备阶段冻结的 ROI，所以边界、缓冲尺寸和消费范围一致。普通相机仍使用局部矩形 ROI。

真实跨缝产物为 `rig_side/frame_000/cam4` 的 **5760×2411** 全宽条带。person 1 的目标 mask 在原图两端，最短周期框是 `[5486, 767, 6383, 2206]`（展开的右端超过 5760），而不是近 5760 像素宽的普通框。[连续查看图](../outputs/inspection/city_four_first_20261024/cycle_walk_seam_labels.png)显示人物、mask 边缘及点可接通；[原始 roi.png](../outputs/runs/city_four_first_20261024_cycle_walk_seam_fix2/rigs/rig_side/frame_000/cam4/roi.png)仍按原 ERP 保存。

消费端的深度、实例 ID 与 3×3/5×5 可见性邻域均按横向周期访问；未渲染的位置保持 unknown，不从黑色 RGB 推断可见。目标裁剪只在全宽条带使用周期取样，垂直方向不环绕。保存的有序 512 点仍是**原始完整 ERP XY**；reader 在映射到 384×512 裁剪图时临时展开 x，坐标还原时取模，ERP 评价横向误差使用最短周期距离。此目标 512 点均有坐标监督，visibility 有 172 个 `1`、0 个 `0`、340 个 `-1`。

数据包顶层仍是 `mamma_bundle_v1`，标签 NPZ 的坐标和 visibility 字段未改。观察行中的 `crop_support.crop_policy_version` 明确区分 `mask_bbox_periodic_v2` 与普通的 `mask_bbox_inclusive_v1`，并保存展开后的框和采样范围。MAMMA 当前 reader 配置升至 `mamma_reader_v2`、训练配置升至 `mamma_training_v3`、恢复合同升至 `mamma_regular_resume_v4`；原有非跨缝数据仍按行中旧裁剪策略读取，没有改写历史包。

## 回读与画面判断

四包的 MAMMA `plan` 均通过，未构造模型；每类每人至少实读一条 observed（共 8 人次）。跨缝目标的 `img` 为 3×512×384、mask 为 1×512×384，512 个坐标均有效；把其 `xy_target` 还原到原 ERP 后最大差约 **0.00015 px**。小夹具还对照了周期平移前后的 RGB/mask 裁剪、靠近 x=0/W 的点与邻域，非跨缝回归通过。生成器当前定向测试 23 项、MAMMA 8 项通过；原场景重新 CPU prepare 的 frame 0/cam2 与 cam4 ROI 和真实任务相同。CPU 与 Blender 用同一冻结计划，实际全宽 PNG 尺寸与计划一致。

画面里骑行人物有骑姿，**没有自行车道具**。用户已明确本项目只需要骑行姿态、不制作自行车，因此这符合本批目标，不作为扩批前置问题。三人场景有人物仅约 61×127 像素；一人骑车场景的代表目标约 380×588 像素。骑车＋走路在 frame 14 的 rig_custom_1/cam2 确有近距离交错。[双人交错叠加图](../outputs/inspection/city_four_first_20261024/two_cycle_occlusion_p2.png)中的后方人物有 263 个负标签点，前景实例 ID 全是另一人物，深度差为 0.22–1.17 米（中位 0.76 米）；这提供一例可检查的遮挡证据，**还不能据此宣布全批 visibility 负例资格完成**。训练仍被当前配置禁止，也未执行模型评价。没有扩批、整段渲染或修改规则。
