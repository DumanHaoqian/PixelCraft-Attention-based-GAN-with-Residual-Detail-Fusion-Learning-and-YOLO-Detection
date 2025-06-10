# YOLOv8 TTA + Few-Shot


# %% ------------------------ 0. Multi-version Loader -------------------------------
import importlib, glob, cv2, numpy as np
from pathlib import Path

def _simple_loader(path, batch=1, img_size=640):
    files = sorted(glob.glob(str(Path(path) / "*.*")))
    for i in range(0, len(files), batch):
        sub = files[i:i + batch]
        imgs = []
        for p in sub:
            im = cv2.imread(p)
            im = cv2.resize(im, (img_size, img_size))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.transpose(2, 0, 1)              # HWC→CHW
            imgs.append(im)
        yield sub, np.stack(imgs), None, None, ''

def LoadMedia(path, img_size=640, stride=32, batch=1, vid_stride=1):
    try:  # ≥8.2
        load_inf = importlib.import_module(
            'ultralytics.data.loaders').load_inference_source
        for imgs, paths, _, _ in load_inf(source=path,
                                          imgsz=img_size,
                                          stride=stride,
                                          batch=batch,
                                          vid_stride=vid_stride):
            yield paths, imgs, None, None, ''
        return
    except (ImportError, AttributeError):
        pass
    try:  # 8.1.x
        LIV = importlib.import_module(
            'ultralytics.data.stream_loaders').LoadImagesAndVideos
        for out in LIV(path, img_size, stride, batch, vid_stride):
            yield out
        return
    except (ImportError, AttributeError):
        pass
    try:  # 8.0.x
        LI = importlib.import_module(
            'ultralytics.data.loaders').LoadImages
        for out in LI(path, img_size, stride, batch, vid_stride):
            yield out
        return
    except (ImportError, AttributeError):
        pass

    print("⚠ Official loader not found, using simple loader (images only).")
    yield from _simple_loader(path, batch, img_size)

# %% --------------------- 1. Global Configuration ----------------------------------
import os, time, csv, shutil, warnings
warnings.filterwarnings("ignore")

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

PROJECT_DIR     = Path(__file__).parent.resolve()
DATA_YAML       = PROJECT_DIR / "dataset.yaml"
GEN_IMAGE_DIR   = PROJECT_DIR / "gen_images"
TEST_IMAGE_DIR  = PROJECT_DIR / "test_images"
RESULT_DIR      = PROJECT_DIR / "yolo_results"
BASE_WEIGHTS    = "yolov8n.pt"

IMG_SIZE        = 640
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"

# Few-Shot batch experiments (all with TTA)
EXP_EPOCHS      = [20,50,100]
FEW_SHOT_LR     = 3e-4
FEW_SHOT_BATCH  = 16
NUM_WORKERS     = 4

# TTA (Ultralytics augment=True)
TTA_BATCH = 16

RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f"‣ Using device: {DEVICE}")

# %% --------------------- 2. Utility Functions -------------------------------------
def timer(fn):
    def _wrap(*a, **kw):
        s = time.time()
        out = fn(*a, **kw)
        print(f"[{fn.__name__}] Done in {time.time()-s:.1f}s")
        return out
    return _wrap

def _mp(m): return getattr(m.box, 'mp', getattr(m.box, 'precision', 0.))
def _mr(m): return getattr(m.box, 'mr', getattr(m.box, 'recall',   0.))
def _fps(m):
    spd = getattr(m, 'speed', {})
    return 1000/(sum(spd.values()) or 1) if isinstance(spd, dict) else getattr(spd,'fps',0.)

@timer
def baseline_eval(model: YOLO, tta=False):
    return model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=FEW_SHOT_BATCH,
        split="val",
        device=DEVICE,
        augment=tta,
        verbose=False,
        save_json=False
    )

@timer
def tta_eval(model: YOLO):
    return baseline_eval(model, tta=True)

# —— Few-Shot —— #
def freeze_backbone(m: YOLO):
    for p in m.model.parameters(): p.requires_grad = False
    head = m.model.model[-1] if hasattr(m.model,'model') else m.model[-1]
    for p in head.parameters(): p.requires_grad = True

@timer
def few_shot_finetune(base_w: str, save_pt: Path, epochs: int, tag: str):
    m = YOLO(base_w)
    freeze_backbone(m)
    m.train(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        epochs=epochs,
        batch=FEW_SHOT_BATCH,
        lr0=FEW_SHOT_LR,
        optimizer="AdamW",
        device=DEVICE,
        workers=NUM_WORKERS,
        project=str(RESULT_DIR),
        name=tag,
        close_mosaic=5,
        exist_ok=True,
        verbose=False
    )
    best = Path(m.trainer.save_dir) / "weights" / "best.pt"
    shutil.copy(best, save_pt)
    return save_pt

# —— Visualization —— #
def visualize(model: YOLO, name: str, tta=False):
    if not TEST_IMAGE_DIR.exists():
        print("⚠ test_images directory not found, skipping visualization."); return
    model.predict(
        source=str(TEST_IMAGE_DIR),
        imgsz=IMG_SIZE,
        save=True,
        conf=0.25,
        augment=tta,
        project=f"vis_{name}",
        name="predict",
        exist_ok=True,
        verbose=False
    )

# —— Safe CSV writing —— #
def _safe_csv_writer(path: Path):
    try:
        return open(path, "w", newline="", encoding="utf-8")
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt = path.parent / f"{path.stem}_{ts}{path.suffix}"
        print(f"⚠ Unable to write to {path}, writing to {alt} instead")
        return open(alt, "w", newline="", encoding="utf-8")

# %% --------------------- 3. Main Workflow ----------------------------------------
def main():
    if DEVICE.startswith("cuda"):
        torch.cuda.set_device(int(DEVICE.split(":")[-1]))

    base_model = YOLO(BASE_WEIGHTS).to(DEVICE)
    print("==> Pretrained weights loaded")

    print("\n==== Baseline Evaluation ====")
    m_base = baseline_eval(base_model, tta=False)
    visualize(base_model, "baseline", tta=False)

    print("\n==== TTA Evaluation ====")
    m_tta = tta_eval(base_model)
    visualize(base_model, "tta", tta=True)

    csv_path = RESULT_DIR / "results.csv"
    with _safe_csv_writer(csv_path) as f:
        wr = csv.writer(f)
        wr.writerow(["mode","epochs","mAP50","mAP50-95","precision","recall","fps"])
        def _write(tag, ep, m):
            wr.writerow([tag, ep, m.box.map50, m.box.map, _mp(m), _mr(m), _fps(m)])

        _write("baseline",0,m_base)
        _write("tta",     0,m_tta)

        # —— Few-Shot multiple runs —— #
        for ep in EXP_EPOCHS:
            tag = f"fewshot_ep{ep}"
            pt  = RESULT_DIR / f"{tag}_best.pt"

            print(f"\n==== Few-Shot Fine-Tune ({ep} epoch) ====")
            if not pt.exists():
                few_shot_finetune(BASE_WEIGHTS, pt, ep, tag)

            ft_model = YOLO(str(pt)).to(DEVICE)

            # Standard validation
            m_ft = baseline_eval(ft_model, tta=False)
            _write(tag, ep, m_ft)
            visualize(ft_model, tag, tta=False)

            # TTA validation
            m_ft_tta = baseline_eval(ft_model, tta=True)
            _write(f"{tag}_tta", ep, m_ft_tta)
            visualize(ft_model, f"{tag}_tta", tta=True)

    # —— Summary print —— #
    print("\n=========== Summary (mAP50 / mAP50-95 / P / R / FPS) ===========")
    fmt = lambda m: f"{m.box.map50:.4f}, {m.box.map:.4f}, {_mp(m):.4f}, {_mr(m):.4f}, {_fps(m):.2f}"
    print(f"Baseline        : {fmt(m_base)}")
    print(f"TTA             : {fmt(m_tta)}")
    for ep in EXP_EPOCHS:
        res = RESULT_DIR / f"fewshot_ep{ep}_best.pt"
        if res.exists():
            m1 = baseline_eval(YOLO(str(res)).to(DEVICE), tta=False)
            m2 = baseline_eval(YOLO(str(res)).to(DEVICE), tta=True)
            print(f"Few-Shot-{ep:>3}       : {fmt(m1)}")
            print(f"Few-Shot-{ep:>3}+TTA   : {fmt(m2)}")

    print(f"\nResults saved to: {csv_path.parent.resolve()}")

# %% --------------------- 4. Execution -------------------------------------------
def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
elif _in_notebook():
    main()