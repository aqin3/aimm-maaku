#!/usr/bin/env python3
# Train YOLO for buoy color detection with YAML configs.
# Requires: pip install ultralytics==8.*  (or your preferred v8/v9 build)

import torch
from ultralytics import YOLO

def main():
    # Fast dataloader and kernel autotune on 4080
    torch.backends.cudnn.benchmark = True

    # Pick a model size. For a good balance, start with yolov8s; try yolov8m if data is large.
    # You can swap to a custom .pt if you have one.
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="data_buoys.yaml",     # dataset YAML below
        cfg=None,                   # keep None unless you use a model cfg
        epochs=100,
        imgsz=640,                  # 512 if you prefer speed over accuracy
        batch=-1,                   # auto-batch for 4080
        device=0,                   # 0 = first GPU
        workers=12,                 # 8–16 is fine on most CPUs
        seed=42,
        project="runs_buoy",
        name="y8s_640_e100",
        verbose=True,
        save=True,
        save_period=0,              # set >0 to checkpoint every N epochs
        patience=50,                # early stop if val plateaus
        optimizer="AdamW",          # good default for 4080
        lr0=0.0025,                 # AdamW base LR
        lrf=0.1,                    # final LR factor
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,                # cosine schedule
        hsv_h=0.02,                 # safe if you skip hyp file; better to use hyp YAML below
        hsv_s=0.35,
        hsv_v=0.30,
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        translate=0.10,
        scale=0.60,
        shear=0.10,
        perspective=0.0005,
        mosaic=0.20,
        mixup=0.0,
        copy_paste=0.10,
        # If you prefer managing all aug/optim in YAML, comment the aug lines above and pass hyp:
        # hyp="hyp_buoys.yaml",
        pretrained=True,
        cache=True,                 # cache images to RAM
        persist=True,               # persistent workers
        amp=True                    # mixed precision for speed
    )

    print("Training complete. Best weights:", results.best)

    # Optional: validate the best checkpoint on the test split defined in data_buoys.yaml
    model = YOLO(results.best)
    model.val(data="data_buoys.yaml", split="test", imgsz=640, device=0, workers=12)

    # Optional: export for deployment (pick one)
    # model.export(format="onnx", half=True)
    # model.export(format="engine", half=True)  # TensorRT FP16
    # model.export(format="torchscript", half=True)

if __name__ == "__main__":
    main()

