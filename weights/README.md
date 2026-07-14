# Model Weights

Place trained or exported model files here.

Expected paths:

```text
weights/yolo26s-pose.onnx
weights/yolo26s-pose.engine
weights/video_swin_tiny.onnx
weights/video_swin_tiny.engine
```

The code also accepts these YOLO26 pose aliases without config changes:

```text
weights/yolo26-pose.onnx
weights/yolo26-pose.engine
```

Model sources:

- YOLO26-Pose: https://github.com/ultralytics/ultralytics
- ByteTrack: https://github.com/FoundationVision/ByteTrack

ByteTrack is used as a tracking algorithm and does not require a weight file in this project.

- Video Swin Transformer: https://github.com/SwinTransformer/Video-Swin-Transformer

Weight files are intentionally ignored by `.gitignore`.

The supplied Video Swin model must output three logits in this exact order:

```text
0 normal
1 fight
2 fall
```

The current ONNX export uses input shape `1x1x3x32x224x224`. Keep the
external data file named exactly as referenced by the ONNX graph (currently
`epoch_37.onnx.data`) in the same directory as `video_swin_tiny.onnx`.
