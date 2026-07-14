# DBehavior Edge 技术交接报告

文档版本：2026-07-13  
适用代码：第一阶段端侧闭环  
目标硬件：Jetson Orin NX 16GB  
主配置：`configs/edge_orin.yaml`

## 1. 阅读目标

本文面向下一位接手代码的算法或部署工程师。读完后应能够：

1. 解释一帧视频如何经过检测、跟踪、动作识别、融合和状态机形成告警。
2. 明确两个深度学习模型的输入输出契约，替换权重时知道检查什么。
3. 根据诊断文件定位“没有检测到”“有概率但不触发”“误报过多”等问题。
4. 在不破坏现有闭环的前提下增加行为类别、替换模型、接入平台回调或迁移 TensorRT。
5. 知道当前实现有哪些已验证能力，哪些仍是第一阶段技术债。

## 2. 当前结论

项目已经形成可运行的单路端侧闭环：

```text
RTSP/MP4
  -> YOLO26s-Pose
  -> ByteTrack
  -> Person/Group ROI
  -> 32-frame Clip
  -> Video Swin-Tiny
  -> 规则融合
  -> 事件状态机
  -> JSONL + 截图 + 前后证据视频
```

当前 Windows `x-gpu` Conda 环境已使用 ONNX 权重完成真实视频回归，模型没有进入降级模式。测试视频 `1329724835.mp4` 的最近一次结果：

| 项目            | 结果                      |
| ------------- | -----------------------:|
| 解码原始信息        | 1280x720, 30 FPS, 474 帧 |
| 处理帧           | 237（采样到 15 FPS）         |
| 检测累计          | 304                     |
| 跟踪累计          | 285                     |
| Fall ROI      | 268                     |
| Fight ROI     | 85                      |
| 动作 Clip       | 52                      |
| 最大 `P(fight)` | 0.9999996               |
| 最大 Fight 融合分数 | 0.8999957               |
| Fight 确认数     | 1                       |
| 回退通道确认时刻      | 视频时间 5.20 s             |

告警截图位于 `docs/assets/fight_detection_sample.jpg`。画面中两人发生严重遮挡，Pose 只保留一个合并框，单人回退通道正确触发。

注意：以上是功能与数值链路验证，不是 Orin NX 性能验收结果。

## 3. 设计范围

### 3.1 已实现

- 单路 RTSP 或本地视频输入。
- H.264 RTSP 的 Jetson GStreamer 硬解码管线。
- YOLO26s-Pose ONNX/TensorRT 接口和 Ultralytics Pose 输出解析。
- ByteTrack 检测框关联，同时携带关键点历史。
- 单人摔倒 ROI、双人打架 Group ROI。
- Video Swin-Tiny 三分类动作推理。
- 模型概率与姿态/运动几何特征融合。
- Fight/Fall 状态机、冷却去重。
- 截图、证据视频、JSONL、可视化、逐窗口诊断。
- dry-run 模型和自动化测试。

### 3.2 尚未实现

- 多路视频统一调度和 GPU 资源治理。
- HTTP、MQTT、GB28181 平台回调。
- 事件按人员/群组实体并行维护。
- 配置热更新、服务守护、健康检查和指标采集。
- 日志/JSONL 自动轮转。
- TensorRT 10 name-based API 兼容和动态 shape。
- Orin NX 上正式的 FPS、P95 延迟、显存与 30 分钟稳定性报告。
- 模型训练、标注、评估工具链与端侧仓库的一体化版本管理。

## 4. 数据流与控制流

```mermaid
sequenceDiagram
    participant V as VideoSource
    participant P as PoseInferencer
    participant T as ByteTrack
    participant C as Candidate + ClipBuilder
    participant A as ActionInferencer
    participant F as Fusion
    participant S as StateMachine
    participant O as EventWriter

    V->>P: FramePacket(frame_id, timestamp, BGR)
    P->>T: PersonDetection[]
    T->>C: Track[] + history
    C->>C: build person/group ROI and accumulate frames
    C->>A: N x 32 x 224 x 224 x 3 RGB
    A->>F: normal/fight/fall probabilities
    T->>F: geometry and motion features
    F->>S: EventCandidate[]
    S-->>O: CONFIRMED EventRecord
    O->>O: JSONL + snapshot + evidence video
```

主编排位于 `edge_app/pipeline.py::BehaviorPipeline`。修改任何阶段时，优先保持以下接口不变：

```python
PoseInferencer.infer(frame_bgr) -> list[PersonDetection]
ByteTrack.update(detections, timestamp) -> list[Track]
ClipBuilder.add(packet, rois) -> tuple[list[CandidateROI], np.ndarray]
ActionInferencer.infer(clips_rgb) -> list[ActionScore]
EventStateMachine.update(candidates, timestamp) -> list[EventRecord]
```

## 5. 模块职责与修改入口

| 文件                          | 责任                       | 常见修改                        |
| --------------------------- | ------------------------ | --------------------------- |
| `edge_app/run.py`           | CLI 参数、配置加载、进程返回码        | 新增 CLI 参数、服务入口              |
| `edge_app/config.py`        | YAML 对应 dataclass        | 新增配置项及默认值                   |
| `edge_app/video.py`         | RTSP/文件解码、采样、时间戳         | H.265、重连策略、多协议              |
| `edge_app/runtime.py`       | ONNX Runtime/TensorRT 会话 | TRT 10、动态 shape、CUDA stream |
| `edge_app/inferencers.py`   | 模型前后处理                   | 替换 YOLO/Video Swin、量化输入     |
| `edge_app/tracker.py`       | ByteTrack/Kalman/关联      | ReID、GMC、遮挡关联优化             |
| `edge_app/geometry.py`      | ROI 和姿态几何特征              | 新摔倒特征、尺度归一化                 |
| `edge_app/candidates.py`    | 行为候选与 ROI 构建             | 新行为候选、多人聚类                  |
| `edge_app/clips.py`         | ROI 时序队列和采样              | 时间对齐、批处理、缺帧策略               |
| `edge_app/fusion.py`        | 概率与规则加权                  | 标定模型、学习式融合                  |
| `edge_app/events.py`        | 状态转移和告警去重                | 按实体状态、多级告警                  |
| `edge_app/buffer.py`        | 证据帧环形缓存                  | 内存优化、编码前缓存                  |
| `edge_app/output.py`        | JSONL、截图和视频              | 回调、异步写盘、编码器                 |
| `edge_app/diagnostics.py`   | 运行统计和逐窗口日志               | Prometheus、采样与轮转            |
| `edge_app/visualization.py` | 调试叠加                     | 显示原始概率和状态                   |
| `edge_app/types.py`         | 跨模块数据结构                  | 新字段必须同步输出/测试                |

## 6. 模型契约

### 6.1 YOLO26s-Pose

配置节点：`models.pose`

默认模型路径：

```text
weights/yolo26s-pose.engine
weights/yolo26s-pose.onnx
```

预处理：

1. BGR 原图 letterbox 到 `640x640`，填充值 114。
2. BGR 转 RGB。
3. 转 FP32 并除以 255。
4. HWC 转 NCHW，形成 `1x3x640x640`。

后处理输出为：

```python
PersonDetection(
    bbox_xyxy=(x1, y1, x2, y2),
    score=float,
    keypoints=np.ndarray(shape=(17, 3)),  # x, y, confidence
)
```

`_parse_ultralytics_pose_outputs` 支持单输出预测矩阵和 boxes/scores/keypoints 多输出形式。替换 Pose 权重后必须先打印真实输出 shape，并确认：

- 框是 `xywh` 还是 `xyxy`。
- 坐标是否已归一化。
- 置信度列位置是否仍为第 5 列。
- 关键点是否为 COCO 17 点及相同顺序。
- 导出是否包含 NMS。

`box_format: auto` 通过启发式判断格式。生产模型明确时建议直接写 `xywh` 或 `xyxy`，减少误判风险。

### 6.2 ByteTrack

ByteTrack 不是第二个检测模型，不需要权重。它消费 Pose 模型的人员框和置信度，使用 Kalman Filter、IoU 和高低置信度两阶段匹配维持 `track_id`。

当前关键参数：

```yaml
tracking:
  track_high_thresh: 0.40
  track_low_thresh: 0.10
  match_thresh: 0.75
  max_age: 30
  min_hits: 2
```

历史上 `match_thresh=0.30` 过严，打架快速运动时 track id 频繁断裂，Group ROI 无法累计满动作 Clip。真实视频回归表明调整为 `0.75` 后 pair `1-6` 能稳定形成 5 个动作窗口。

环境安装 `lap` 时使用 LAPJV；未安装时回退到贪心匹配。两者在密集人群中的结果可能不同，部署环境必须记录是否安装 `lap`。

### 6.3 Video Swin-Tiny

配置节点：`models.action`

当前 ONNX 实际契约：

```text
input  : [1, 1, 3, 32, 224, 224]
output : [1, 3]
labels : [normal, fight, fall]
```

类别顺序来自预训练数据：

```text
normal -> 0
fight  -> 1
fall   -> 2
```

训练侧配置来源：

```text
Video-Swin-Transformer/configs/recognition/swin/
swin_tiny_patch244_window877_behavior3cls.py
```

训练采样为 `clip_len=32, frame_interval=2`。多数训练视频约为 30 FPS，因此覆盖约 2.1 秒。端侧输入先降为 15 FPS，再使用 `32x1`，保持接近的时间跨度。

端侧预处理：

1. 根据 person/group bbox 裁剪并 letterbox 到 `224x224`。
2. BGR 转 RGB。
3. 除以 255。
4. 使用 ImageNet mean/std 标准化。
5. `N,T,H,W,C -> N,C,T,H,W`。
6. 对六维导出增加 `num_clips=1` 维度。

代码会强制检查 logits 数量等于 labels 数量。替换动作模型时，不允许通过截断或补零绕过该检查。

当前 ONNX 使用 external data：图文件名为 `video_swin_tiny.onnx`，但内部引用 `epoch_37.onnx.data`。该文件必须保留原名并与 ONNX 放在同一目录。重新导出后应使用 ONNX 工具检查 external data location，避免只重命名主图。

### 6.4 当前预训练模型评估背景

现有 epoch 37 测试结果来自训练仓库，不由本端侧仓库生成：

| 指标                  | 数值     |
| ------------------- | ------:|
| Top-1 Accuracy      | 0.7735 |
| Mean Class Accuracy | 0.7712 |
| Fight Precision     | 0.8636 |
| Fight Recall        | 0.7215 |
| Fall Precision      | 0.6752 |
| Fall Recall         | 1.0000 |

这意味着：Fight 模型存在漏检风险；Fall 模型召回很高但误报倾向较强。因此当前 Fight 使用连续 Clip 确认，Fall 保留较强几何约束。阈值不能只依据训练 accuracy 决定，必须在端侧 ROI 分布上重新标定。

## 7. 视频采样与时间戳

### 7.1 本地文件

本地文件按 `input.target_fps` 采样，时间戳来自 `CAP_PROP_POS_MSEC`，不可用时回退到 `decoded_frame_id/source_fps`。因此事件时间是视频内相对秒数，例如 `confirm_ts=5.2`。

### 7.2 RTSP

RTSP 使用 `time.time()`，事件时间为 Unix epoch 秒。GStreamer appsink 设置 `drop=true sync=false`，目标是低延迟而非逐帧完整。

下游平台必须知道本地文件时间和 RTSP 时间语义不同。若接口要求统一，建议给 `FramePacket` 增加 `timestamp_kind`，并同时记录 UTC 时间和 media PTS。

### 7.3 当前 GStreamer 管线

```text
rtspsrc latency=120
  -> rtph264depay
  -> h264parse
  -> nvv4l2decoder
  -> nvvidconv
  -> BGR appsink
```

H.265 需要改为 `rtph265depay ! h265parse`。编码类型自动探测尚未实现。

## 8. 候选 ROI 与 Clip

### 8.1 Fall Person ROI

每条稳定 Track 计算：

```text
fall_hint = 0.35 * vertical_drop
          + 0.35 * body_angle
          + 0.30 * low_posture
```

`fall_hint >= 0.20` 或 history 达到 4 帧时创建 person ROI。后一个条件是为了让动作模型即使在几何特征不明显时仍获得序列输入。

### 8.2 Fight Group ROI

对 Track 两两组合：

```text
fight_hint = 0.45 * proximity
           + 0.35 * limb_motion
           + 0.20 * overlap
```

满足 `fight_hint >= 0.10` 且 `proximity >= 0.10` 时，合并两人框并放大 1.45 倍形成 group ROI。

当前复杂度是 `O(number_of_tracks^2)`。普通路面场景可接受，密集人群应先做空间网格或聚类，限制 pair 数量，否则动作模型调用数会快速增长。

### 8.3 ClipBuilder

队列 key：

```text
{event_type}:{roi_type}:{sorted_track_ids}
```

默认累计 32 帧，每 4 个处理帧输出一次滑动窗口。ROI 连续缺失超过约两个窗口后清理队列。

重要限制：队列只在 ROI 出现时追加，因此中间短暂丢失 ROI 后，前后帧会被直接拼接，时间间隔不再均匀。下一阶段可将队列改为按 timestamp 重采样，并对缺帧选择重复上一帧、插值或整段作废。

## 9. 分数融合

### 9.1 Fight

```text
fight_score = 0.72 * P(fight)
            + 0.10 * proximity
            + 0.08 * limb_motion
            + 0.05 * overlap
            + 0.05 * continuous
```

权重和为 1。当前模型占主导，几何特征用于增强可信度而不是硬判定。

遮挡时两人可能被 Pose 合并成一个框。Person ROI 的 `P(fight) >= 0.85` 时启用单人回退：proximity 和 overlap 置 0，仍使用 motion、track stability 和相同事件状态机。真实样例中该分支在第 5.20 秒确认，截图能看到两人被框在同一检测框内。

### 9.2 Fall

```text
fall_score = 0.55 * P(fall)
           + 0.18 * vertical_drop
           + 0.14 * body_angle
           + 0.08 * low_posture
           + 0.05 * track_stability
```

`vertical_drop` 使用近期 bbox 中心下移量；`body_angle` 使用肩与髋连线的水平程度；`low_posture` 使用 bbox 宽高比；所有特征裁剪到 `[0,1]`。

这些规则对云台运动并不完全鲁棒。机器人快速俯仰时，画面整体运动会影响 drop 和 keypoint motion。后续应接入 IMU/云台姿态，或先做全局运动补偿。

## 10. 事件状态机

状态机按 `fall` 和 `fight` 各维护一份全局状态：

```mermaid
stateDiagram-v2
    NORMAL --> SUSPECT: score >= suspect_threshold
    SUSPECT --> NORMAL: score < suspect_threshold
    SUSPECT --> NORMAL: no candidate > candidate_gap_sec
    SUSPECT --> CONFIRMED: confirm condition
    CONFIRMED --> COOLDOWN: emit EventRecord
    COOLDOWN --> NORMAL: timestamp >= cooldown_until
```

Fight 默认参数：

```yaml
suspect_threshold: 0.38
confirm_threshold: 0.52
confirm_clip_count: 3
candidate_gap_sec: 1.0
cooldown_sec: 10.0
```

Fall 默认参数：

```yaml
suspect_threshold: 0.48
confirm_threshold: 0.62
confirm_duration_sec: 0.8
candidate_gap_sec: 1.0
cooldown_sec: 8.0
```

Fight 的第一次候选进入 SUSPECT 并将 `clip_count=1`；后续候选达到 confirm threshold 且计数达到 3 时确认。Fall 使用 `candidate.timestamp - start_ts` 确认持续时长。

### 10.1 已知状态机技术债

`states` 当前只以行为类型为 key，而不是 `(event_type, entity_key)`。结果是：

- 同一画面多个独立打架群组不能同时维护状态。
- 连续候选可能来自不同 track pair，理论上可能累加到同一确认计数。
- 冷却期内该视频的所有同类型事件都会被抑制。

建议升级路径：

1. Fight entity key 使用排序后的 track id tuple，并增加短期 group id 合并。
2. Fall entity key 使用单个 track id。
3. `states: Dict[Tuple[event_type, entity_key], _BehaviorState]`。
4. 对失效实体设置 TTL 清理。
5. EventWriter 和诊断保留 entity key。

这是扩展到多人复杂场景前优先级最高的算法改造之一。

## 11. 证据输出

环形缓存时长：

```text
evidence_pre_sec + evidence_post_sec + 2 seconds
```

事件确认后先进入 pending，等待 post 时间结束再写盘。视频结束或进程正常退出时 `flush_pending()` 会立即写出剩余事件，因此尾部事件可能不足完整 post 时长。

`events.jsonl` 字段：

| 字段              | 含义                      |
| --------------- | ----------------------- |
| `event_id`      | `{type}-{12-char uuid}` |
| `event_type`    | `fight` 或 `fall`        |
| `state`         | 固定为 `CONFIRMED`         |
| `start_ts`      | 进入 SUSPECT 时间           |
| `confirm_ts`    | 确认时间                    |
| `score`         | SUSPECT 期间最大融合分数        |
| `track_ids`     | 确认候选关联的 track id        |
| `bbox_xyxy`     | 确认候选区域                  |
| `snapshot_path` | 截图路径                    |
| `video_path`    | 证据视频路径                  |
| `source`        | 输入源原文                   |

安全注意：`source` 对 RTSP 会包含完整 URL，可能带用户名和密码。生产环境应在写日志和事件前对凭据脱敏。

## 12. 诊断与阈值标定

### 12.1 诊断文件

`diagnostics.json`：

- frames、detections、tracks。
- fall/fight ROI 数量。
- action batch 和 clip 数量。
- 每类最大/平均概率。
- 每类最大融合分数。
- confirmed 事件数量。

`action_scores.jsonl` 每行包含：

- frame id、timestamp。
- ROI 类型、track ids、bbox。
- 候选几何 features。
- 原始三分类 probabilities。
- fused score 和状态机快照。

注意：每次启动会清空 `action_scores.jsonl`，但 `events.jsonl` 是追加写入。

### 12.2 “打架没有触发”排障顺序

严格按上游到下游排查：

1. `detections == 0`：检查 Pose 输出解析、阈值、输入尺度和画面质量。
2. detections 有但 `tracks == 0`：检查 high threshold、min hits 和 ByteTrack 关联。
3. tracks 有但 `fight_rois == 0`：检查双人距离、track id 稳定性和 candidate threshold。
4. fight ROI 有但 `ready_clips == 0`：检查是否稳定累计 32 帧、ROI key 是否频繁变化。
5. clip 有但 `P(fight)` 低：检查类别顺序、预处理、时间跨度和 ROI 分布偏移。
6. `P(fight)` 高但 fused score 低：检查权重、single-person gate 和几何特征。
7. fused score 高但无事件：检查 suspect/confirm threshold、连续 clip 数、candidate gap 和 cooldown。
8. 事件有但无证据：检查输出目录权限、OpenCV VideoWriter 和磁盘空间。

### 12.3 阈值标定方法

不要用单条视频拍脑袋调阈值。建议：

1. 构建至少包含 fight、fall、normal、拥抱、搀扶、跑动、弯腰、相机运动的端侧回放集。
2. 保存所有 `action_scores.jsonl` 并关联真值时间段。
3. 分别评估模型概率、融合分数和最终事件级 precision/recall。
4. 先固定模型权重，搜索 fusion weights；再搜索 suspect/confirm 和连续窗口数。
5. 以“每小时误报数 + 事件召回率 + 平均确认延迟”作为部署指标。
6. 为不同摄像头视角保留配置版本，不要覆盖唯一配置。

建议新增 `configs/profiles/`，例如：

```text
configs/profiles/
├── edge_orin_default.yaml
├── edge_orin_low_false_alarm.yaml
└── edge_orin_high_recall.yaml
```

## 13. 推理后端

### 13.1 ONNX Runtime

运行时按可用 provider 选择：优先 CUDA，其次 CPU。当前 `runtime.device` 并未传入 provider 选择逻辑，`runtime.fp16` 也不会自动将 FP32 ONNX 转为 FP16；这两个配置目前是部署意图字段，不是实际强制开关。

后续应将 provider、provider options、线程数和 graph optimization 显式配置化，并在启动日志打印最终 provider。

### 13.2 TensorRT

当前封装特点：

- 单输入。
- 静态 binding shape。
- PyCUDA pagelocked host memory。
- `execute_async_v2`。
- legacy `num_bindings/get_binding_shape` API。

TensorRT engine 必须在目标 JetPack/TensorRT/GPU 架构上生成。不能假定从 x86 或另一版本 JetPack 复制的 engine 可用。

TensorRT 10 迁移重点：

1. 使用 `num_io_tensors` 和 tensor name API。
2. 设置动态输入 shape 后再查询 context tensor shape。
3. 使用 `set_tensor_address` 和 `execute_async_v3`。
4. 区分 host/device location 与输入输出 mode。
5. 增加 engine metadata，记录模型 hash、TRT、CUDA、JetPack 和精度。

## 14. 常见改造指南

### 14.1 替换 Pose 模型

1. 保持 `PersonDetection` 输出契约。
2. 在 `inferencers.py` 增加独立 parser，不要把新格式硬塞进旧启发式。
3. 用固定图片验证 bbox 反变换和关键点坐标。
4. 跑 `tests/test_tracker.py`，再跑真实视频检查 track id。
5. 更新 `weights/README.md` 和主 README 的输入输出 shape。

### 14.2 替换或重新训练 Video Swin

1. 固化 label map 并提交到仓库，禁止只依赖训练目录口头约定。
2. 检查导出输入 rank、T/H/W 和输出 logits 数量。
3. 对齐 RGB/BGR、mean/std、resize/crop 和 frame interval。
4. 用同一 clip 比较 PyTorch 与 ONNX logits。
5. 更新 `models.action.labels/clip_length/sample_stride`。
6. 重新做端侧 ROI 数据标定，不能只看全帧测试集指标。

### 14.3 增加新行为类别

当前代码不是完全配置驱动，需要同时修改：

1. `models.action.labels` 与模型输出。
2. `CandidateROI` 构建逻辑。
3. `fusion.py` 的行为分数。
4. `EventStateMachine.states` 和 `_cfg`。
5. `EventsConfig` 对应配置。
6. 可视化颜色与输出测试。
7. 新行为的状态确认语义：持续时间、窗口数或其他条件。

建议下一步抽象 `BehaviorPlugin` 接口，将 candidate、fusion、confirmation policy 组合起来，避免每增加一个行为都修改多个 `if/elif`。不要在只有 fight/fall 时过早做复杂插件系统；在加入第三个真实行为时重构最合适。

### 14.4 改为按实体并行状态机

优先修改 `events.py`，并为候选增加稳定 `entity_key`。注意 ByteTrack id 在长遮挡后可能变化，因此 fight group 不能简单永久绑定 pair id，需基于空间和时间合并 group identity。

### 14.5 增加 HTTP/MQTT 回调

不要在主推理线程直接发网络请求。建议：

1. EventWriter 落本地作为可靠源。
2. 将 EventRecord 放入有界队列。
3. 独立 worker 执行回调、重试和签名。
4. 使用 event_id 保证幂等。
5. 记录 delivered/failed 状态，但不要改写历史 JSONL。

### 14.6 多路视频

每路流必须有独立的 tracker、clip buffer、state machine 和 evidence buffer。模型 session 可以考虑共享，但 TensorRT context 和 CUDA stream 的线程安全需要单独设计。不要直接让多个线程并发调用当前 `TensorRTRuntimeSession`。

## 15. 已知风险清单

| 优先级 | 风险                         | 影响                  | 建议                          |
| --- | -------------------------- | ------------------- | --------------------------- |
| P0  | TensorRT 尚未在目标 Orin 验收     | 无法承诺实时性             | 在目标 JetPack 生成 engine 并压测   |
| P0  | RTSP URL 原样写入事件            | 凭据泄露                | 增加 URL 脱敏                   |
| P1  | 状态机按行为全局维护                 | 多事件串扰               | 改为 entity-keyed states      |
| P1  | Clip 短暂缺帧后直接拼接             | 时序分布漂移              | timestamp 重采样/作废窗口          |
| P1  | TensorRT legacy 静态 binding | TRT 10/动态 shape 不兼容 | name-based API 适配           |
| P1  | 诊断 JSONL 无轮转               | 长期运行占满磁盘            | 大小/时间轮转                     |
| P1  | H.264 管线写死                 | H.265 RTSP 失败       | 编码配置或自动探测                   |
| P2  | Pair 枚举 O(N^2)             | 密集场景负载高             | 空间聚类和 ROI 上限                |
| P2  | Evidence 写盘同步执行            | 告警时阻塞推理             | 异步编码队列                      |
| P2  | 云台运动影响几何特征                 | Fall/Fight 误报       | IMU/GMC 运动补偿                |
| P2  | `device/fp16` 配置未真正控制后端    | 配置认知偏差              | 接入 runtime provider/options |

## 16. 测试与验收

### 16.1 当前自动化测试

```bash
python -m pytest -q
```

当前基线：`9 passed`。

测试覆盖：

- bbox 扩展、裁剪、边界处理。
- Clip 采样与滑窗。
- Group ROI 构建。
- ByteTrack id 连续性。
- Fight/Fall 状态转移和冷却。
- 单人高置信 Fight 回退。
- 告警 JSONL、截图和视频。
- dry-run 主流水线。

### 16.2 修改后的最低回归要求

每次修改至少执行：

```bash
python -m pytest -q
python -m compileall -q edge_app
```

涉及模型前后处理、跟踪、候选或融合时，必须再跑固定视频：

```powershell
conda run -n x-gpu python -m edge_app.run `
  --config configs/edge_orin.yaml `
  --source "dataset\1329724835.mp4" `
  --save-vis
```

核对：

- `action_degraded == false`。
- detections/tracks/fight_rois/ready_clips 不为 0。
- `P(fight)` 与 fused score 分布没有异常漂移。
- 产生一条 CONFIRMED fight 事件。
- 截图 bbox 覆盖真实行为区域。
- 证据视频可播放且包含告警前后内容。

### 16.3 Orin NX 验收表

| 项目   | 验收建议                                |
| ---- | ----------------------------------- |
| 输入   | 720p H.264 RTSP @ 15 FPS            |
| 连续运行 | 至少 30 分钟                            |
| 正确性  | 与 ONNX 固定 clip 的 top-1 一致，概率误差在约定范围 |
| 实时性  | 记录端到端平均/P95 延迟和实际处理 FPS             |
| 资源   | 记录显存峰值、RAM 趋势、CPU/GPU 利用率、温度和功耗模式   |
| 网络   | 断流、重连、抖动和错误 URL                     |
| 存储   | 告警突发、磁盘满、目录无权限                      |
| 稳定性  | 无持续内存增长、无 CUDA/TensorRT context 泄漏  |

性能目标应由产品场景确认后写入，当前代码不应虚构 FPS 数字。

## 17. GitHub 与仓库卫生

`.gitignore` 已排除：

- `dataset/`
- `outputs/`
- ONNX、TensorRT engine、PyTorch 权重和 external data
- Python cache、pytest cache

上传前执行：

```bash
git status --short
git check-ignore -v weights/video_swin_tiny.onnx
git check-ignore -v weights/epoch_37.onnx.data
git check-ignore -v dataset/1329724835.mp4
python -m pytest -q
```

注意：如果大文件曾经被 git add/commit，仅添加 `.gitignore` 不会从历史移除，需要使用合适的历史清理工具。执行历史重写前必须先备份并与协作者确认。

公开发布还需要：

1. 增加本项目 LICENSE。
2. 核对 Ultralytics、ByteTrack、Video Swin 代码和模型许可证。
3. 增加第三方声明和模型来源。
4. 确认演示图片和训练数据可公开使用。
5. 删除任何 RTSP 地址、账号、内部路径和个人信息。

## 18. 接手后的建议顺序

1. 在现有 Windows 环境复跑 9 个测试和固定视频，建立本机基线。
2. 阅读 `diagnostics.json` 与 `action_scores.jsonl`，理解四个阶段计数。
3. 在 Orin NX 上安装与 JetPack 匹配的运行环境，先跑 ONNX。
4. 分别转换 Pose 与 Action TensorRT engine，做逐模型数值对齐。
5. 完成 30 分钟 RTSP 验收并形成性能报告。
6. 优先修复 RTSP 凭据脱敏和诊断轮转。
7. 再改造 entity-keyed 状态机和 timestamp 对齐 Clip。
8. 最后接平台回调和多路调度。

这个顺序能先锁定模型与部署基线，再处理结构扩展，避免同时改变过多变量导致无法归因。

## 19. 快速命令索引

运行测试：

```bash
python -m pytest -q
```

本地视频：

```bash
python -m edge_app.run --config configs/edge_orin.yaml --source sample.mp4 --save-vis
```

RTSP：

```bash
python -m edge_app.run --config configs/edge_orin.yaml --source 'rtsp://user:password@host/stream'
```

无模型 smoke test：

```bash
python -m edge_app.run --config configs/edge_orin.yaml --source sample.mp4 --dry-run-models --max-frames 100
```

查看最后一个事件（PowerShell）：

```powershell
Get-Content outputs\edge_events\events.jsonl -Tail 1
```

查看诊断：

```powershell
Get-Content outputs\edge_events\diagnostics.json -Raw
```

## 20. 交接检查单

- [ ] 权重文件名和 external data 引用已确认。
- [ ] 动作 label map 已确认并纳入版本管理。
- [ ] `python -m pytest -q` 全部通过。
- [ ] 固定视频 ONNX 回归通过。
- [ ] Orin TensorRT 数值对齐通过。
- [ ] RTSP 30 分钟稳定性通过。
- [ ] 告警截图和证据视频可用。
- [ ] RTSP 凭据已脱敏。
- [ ] 日志和诊断文件已设置轮转。
- [ ] 配置、模型 hash、JetPack/TRT 版本已记录。

接手人若只记住一条原则：**先用诊断数据确定故障位于检测、跟踪、候选、动作模型、融合还是状态机，再修改对应层，避免直接降低最终告警阈值。**
