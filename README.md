# DBehavior Edge

> 面向机器狗云台的端侧人员打架与摔倒识别闭环，目标平台为 Jetson Orin NX 16GB。




DBehavior Edge 将 **YOLO26s-Pose、ByteTrack、Video Swin-Tiny、规则融合与事件状态机** 组合为一条可落地的端侧算法链路。输入支持 RTSP 和本地视频，输出包括结构化 JSONL 告警、截图、前后证据视频、可视化视频以及逐窗口诊断数据。

> 仓库不提交模型权重和数据集。将训练或导出的权重放入 `weights/` 即可运行。

## 效果展示

下图来自真实视频回归。两名人员发生遮挡时，姿态模型一度只输出一个合并框；系统通过高置信度单人回退通道，在视频第 `5.20 s` 确认打架事件。

![Fight detection sample](docs/assets/fight_detection_sample.jpg)

| 指标            | 本地 ONNX 回归结果             |
| ------------- | ------------------------:|
| 输入视频          | 1280x720, 30 FPS, 15.8 s |
| 端侧采样          | 15 FPS, 237 帧            |
| 人员检测累计        | 304                      |
| Fight ROI 累计  | 85                       |
| 动作识别 Clip     | 52                       |
| 最大 `P(fight)` | 1.000                    |
| 最大 Fight 融合分数 | 0.900                    |
| 已确认 Fight 事件  | 1                        |

以上是功能回归结果，不代表 Orin NX 的最终性能指标。部署前应在目标 JetPack、TensorRT 和实际 RTSP 网络环境中完成 30 分钟稳定性及实时性验收。

## 核心能力

- RTSP、本地 MP4 双输入，RTSP 优先使用 Jetson GStreamer 硬解码。
- YOLO26s-Pose 人员框与 17 点姿态推理，支持 ONNX/TensorRT 权重回退查找。
- 内置 ByteTrack 关联逻辑，无需额外 ReID 或跟踪权重。
- Person ROI 与 Group ROI 时序缓存，适配 Video Swin-Tiny 的 32 帧输入。
- `normal / fight / fall` 三分类动作概率与几何、运动特征融合。
- `NORMAL -> SUSPECT -> CONFIRMED -> COOLDOWN` 告警状态机。
- JSONL、截图、前后证据视频、可视化视频与诊断数据完整闭环。
- 无权重 dry-run、模拟 fight/fall、单元测试和本地视频回归路径。

## 算法架构

```mermaid
flowchart LR
    A["RTSP / Local Video"] --> B["Decode + 15 FPS sampling"]
    B --> C["YOLO26s-Pose"]
    C --> D["ByteTrack"]
    D --> E1["Person ROI"]
    D --> E2["Pair / Group ROI"]
    E1 --> F["32-frame Clip Buffer"]
    E2 --> F
    F --> G["Video Swin-Tiny"]
    D --> H["Pose / Motion Geometry"]
    G --> I["Score Fusion"]
    H --> I
    I --> J["Event State Machine"]
    J --> K["JSONL + Snapshot + Evidence Video"]
    I --> L["Diagnostics + Visualization"]
```

打架融合分数默认由以下分量组成：

```text
fight_score = 0.72 * P(fight)
            + 0.10 * proximity
            + 0.08 * limb_motion
            + 0.05 * overlap
            + 0.05 * track_stability
```

摔倒融合分数默认由以下分量组成：

```text
geometry_gate = clip(max(0.25, fall_geometry / 0.15), 0, 1)

fall_score = 0.55 * P(fall) * geometry_gate
           + 0.18 * vertical_drop
           + 0.14 * body_angle
           + 0.08 * low_posture
           + 0.05 * track_stability
```

`fall_geometry` 取 32 帧窗口内姿态摔倒提示的最大值。模型给出高 `P(fall)` 但没有下落、横卧或低姿态证据时，模型项会被软衰减。同一 track/重叠区域刚确认 Fight 后，10 秒内的弱几何 Fall 还会被视为同一冲突事件；强几何摔倒可以覆盖该抑制。

```mermaid
stateDiagram-v2
    [*] --> NORMAL
    NORMAL --> SUSPECT: score >= suspect_threshold
    SUSPECT --> NORMAL: score drops or candidate gap timeout
    SUSPECT --> CONFIRMED: confirmation condition satisfied
    CONFIRMED --> COOLDOWN: write event
    COOLDOWN --> NORMAL: cooldown expires
```

## 快速开始

### 1. 准备环境

Windows 调试环境可直接使用已有 Conda 环境：

```powershell
cd DBehavior
conda run -n x-gpu python -m pytest -q
```

新环境安装基础依赖：

```bash
python -m pip install -r requirements-edge.txt
```

Jetson 上的 TensorRT、PyCUDA、CUDA 和带 GStreamer 的 OpenCV 建议使用 JetPack 自带版本，不建议直接用 PyPI 覆盖系统 TensorRT。

### 2. 放置权重

```text
weights/
├── yolo26s-pose.onnx          # Windows/通用 ONNX 调试
├── yolo26s-pose.engine        # Orin NX 首选
├── video_swin_tiny.onnx
├── epoch_37.onnx.data         # 当前 ONNX 图引用的外部数据文件名
└── video_swin_tiny.engine
```

应用配置默认填写 `.engine`。文件不存在时，会自动查找同名 `.onnx`。当前 Video Swin ONNX 使用外部权重数据，`.onnx.data` 必须与 ONNX 图位于同一目录且保持图中记录的文件名。

### 3. 本地视频运行

```powershell
conda run -n x-gpu python -m edge_app.run `
  --config configs/edge_orin.yaml `
  --source "dataset\1329724835.mp4" `
  --save-vis
```

Linux/Jetson：

```bash
python -m edge_app.run \
  --config configs/edge_orin.yaml \
  --source /data/sample.mp4 \
  --save-vis
```

### 4. RTSP 运行

```bash
python -m edge_app.run \
  --config configs/edge_orin.yaml \
  --source 'rtsp://user:password@192.168.1.10:554/stream'
```

也可将地址写入 `configs/edge_orin.yaml` 的 `input.rtsp_url`，随后省略 `--source`。公开日志或提交 issue 前请移除 RTSP 凭据。

### 5. 无权重链路检查

```bash
python -m edge_app.run \
  --config configs/edge_orin.yaml \
  --source /data/sample.mp4 \
  --dry-run-models \
  --max-frames 100
```

## 模型契约

| 模块              | 输入                       | 输出                  | 默认路径                             |
| --------------- | ------------------------ | ------------------- | -------------------------------- |
| YOLO26s-Pose    | `1x3x640x640`, RGB, FP32 | 人员框、置信度、`17x3` 关键点  | `weights/yolo26s-pose.engine`    |
| ByteTrack       | 人员框、置信度、关键点              | 稳定 `track_id` 和历史轨迹 | 无权重                              |
| Video Swin-Tiny | `1x1x3x32x224x224`, RGB  | 3 logits            | `weights/video_swin_tiny.engine` |

动作类别顺序必须严格一致：

```text
0 normal
1 fight
2 fall
```

代码会校验 logits 数量与 `models.action.labels` 数量；不一致时直接报错，避免错误类别映射被静默用于告警。

## 配置要点

主配置位于 [`configs/edge_orin.yaml`](configs/edge_orin.yaml)。

| 配置                                           | 默认值   | 含义               |
| -------------------------------------------- | -----:| ---------------- |
| `input.target_fps`                           | 15    | 端侧处理帧率           |
| `models.action.clip_length`                  | 32    | 动作窗口帧数           |
| `models.action.infer_interval_frames`        | 4     | 动作推理间隔           |
| `tracking.match_thresh`                      | 0.75  | ByteTrack 关联代价阈值 |
| `events.fight.suspect_threshold`             | 0.38  | Fight 进入可疑状态阈值   |
| `events.fight.confirm_threshold`             | 0.52  | Fight 确认分数阈值     |
| `events.fight.confirm_clip_count`            | 3     | Fight 连续确认窗口数    |
| `events.fight.single_person_model_threshold` | 0.85  | 遮挡合并框回退门槛        |
| `events.fall.model_geometry_gate_threshold`  | 0.15  | Fall 模型项满权重几何门槛  |
| `events.cross_event_suppression_sec`         | 10    | Fight 对弱 Fall 的抑制时间 |
| `events.evidence_pre_sec/post_sec`           | 5 / 5 | 告警前后证据时长         |

阈值调整前先查看诊断文件，不要仅根据单个视频直接降低确认阈值。

## 输出

```text
outputs/edge_events/
├── events.jsonl
├── diagnostics.json
├── action_scores.jsonl
├── snapshots/
├── videos/
└── vis.mp4
```

`events.jsonl` 每行是一条独立事件：

```json
{
  "event_id": "fight-0427c0a9d4c9",
  "event_type": "fight",
  "state": "CONFIRMED",
  "start_ts": 4.6667,
  "confirm_ts": 5.2,
  "score": 0.7785,
  "track_ids": [1],
  "bbox_xyxy": [1095.6, 117.6, 1237.8, 432.5],
  "snapshot_path": "outputs/edge_events/snapshots/fight-....jpg",
  "video_path": "outputs/edge_events/videos/fight-....mp4",
  "source": "sample.mp4"
}
```

`diagnostics.json` 提供全局计数和最大/平均概率；`action_scores.jsonl` 记录每个动作窗口的原始概率、规则特征、融合分数和状态机快照，是排查“不触发”和误报的第一入口。

## 项目结构

```text
DBehavior/
├── configs/edge_orin.yaml       # Orin NX 主配置
├── edge_app/
│   ├── video.py                 # RTSP/本地视频与时间戳
│   ├── inferencers.py           # Pose/Action 前后处理
│   ├── runtime.py               # ONNX Runtime/TensorRT 适配
│   ├── tracker.py               # ByteTrack
│   ├── candidates.py            # Person/Group ROI 候选
│   ├── clips.py                 # 时序 Clip 缓存
│   ├── fusion.py                # 模型与规则融合
│   ├── events.py                # 事件状态机
│   ├── output.py                # 告警证据输出
│   ├── diagnostics.py           # 可观测统计
│   └── pipeline.py              # 主流水线编排
├── tests/                       # 单元与集成测试
├── docs/TECHNICAL_HANDOFF.md    # 接手开发技术报告
├── requirements-edge.txt
└── weights/README.md
```

## 测试

```bash
python -m pytest -q
```

当前覆盖：几何特征、ROI 裁剪、Clip 滑窗、ByteTrack、融合回退、状态机、输出字段和 dry-run 闭环。当前基线为 `9 passed`。

## Orin NX 部署建议

1. 使用与目标 JetPack 完全一致的 TensorRT 版本生成 `.engine`，engine 不保证跨 TensorRT 或 GPU 架构兼容。
2. 先用 ONNX 对齐数值，再分别转换 Pose 和 Action engine，并对同一批输入比较 top-1 与概率误差。
3. RTSP 先确认编码为 H.264；当前 GStreamer 管线使用 `rtph264depay`。
4. 运行 `720p@15FPS`、30 分钟稳定性测试，记录端到端 FPS、P95 延迟、显存、内存和 RTSP 重连次数。
5. 正式部署关闭 `--save-vis`，并对 `action_scores.jsonl` 配置轮转或关闭详细诊断。

## 已知边界与路线图

- 当前事件状态按行为类型全局维护，同一路视频暂不支持多个同类事件并行确认。
- 当前 TensorRT 封装面向静态 shape 和 legacy bindings；TensorRT 10 需要进一步兼容 name-based tensor API。
- GStreamer RTSP 管线当前固定 H.264，H.265 需要切换 depay/parser。
- `runtime.device` 与 `runtime.fp16` 目前是部署配置占位，ONNX provider 和 engine 精度由实际运行时/权重决定。
- 下一阶段建议增加多流调度、HTTP/MQTT 告警回调、日志轮转、Prometheus 指标和 TensorRT 性能基线。

完整设计、修改入口、排障顺序和已知技术债见 [`docs/TECHNICAL_HANDOFF.md`](docs/TECHNICAL_HANDOFF.md)。

## 上游项目

- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)
- [SwinTransformer/Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)

公开发布前请根据上游代码、模型和数据集的许可证补充本仓库许可证及第三方声明。
