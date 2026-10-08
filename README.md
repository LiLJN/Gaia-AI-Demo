# Golf Ball Tracking Demo

`track_video.py` finds the golf ball in a swing video, tracks it after impact, and saves a copy of the video with the ball's path drawn on it.

- **Green line:** the ball was detected in that frame.
- **Yellow line:** the ball's position was estimated (predicted).

## 1. Install

Python 3.10+ is recommended.

```bash
pip install opencv-python numpy sahi rfdetr
```

If your checkpoint is a YOLO model (`.pt`), also run `pip install ultralytics`.

## 2. Get the checkpoint

You need a trained model checkpoint, for example `checkpoint_best_total.pth` (RF-DETR) or `best.pt` (YOLO).

## 3. Run

```bash
python track_video.py <input_video> <output_video> <checkpoint>
```

Example:

```bash
python track_video.py swing.mp4 swing_tracked.mp4 checkpoint_best_total.pth
```

The labelled video is saved to `<output_video>`. The script also prints the estimated launch angle and ball speed.

## Use from Python

```python
from track_video import track_video

metrics = track_video("swing.mp4", "swing_tracked.mp4", "checkpoint_best_total.pth")
print(metrics)  # launch angle / speed, or None if no shot was tracked
```

## Notes

- **GPU:** the script uses a GPU automatically if one is available. Otherwise it runs on the CPU, which works but is much slower.
- **Model size:** the RF-DETR checkpoint must be the **medium** size at resolution **736**. If yours is different, change `RFDETR_SIZE` and `RESOLUTION` at the top of `track_video.py`.
- **Camera view:** the ball should start in the lower part of the frame. The search area is set by `ROI_POLYGON` at the top of `track_video.py`; change it if your camera view is different.
