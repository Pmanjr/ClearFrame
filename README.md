# vcam-clone

Webcam background blur / removal / replacement for Windows that shows up as a
virtual camera in Zoom, Meet, Teams, Discord and OBS. No watermark, and the
person stays genuinely sharp at 1080p.

## The quality rule

The person's pixels are never downscaled.

- Capture is requested at native **1920x1080 MJPG** (USB 2.0 webcams silently
  fall back to 720p or lower in YUY2). The negotiated resolution, FPS and
  FOURCC are logged at startup, with a loud warning if they differ.
- Segmentation runs on a downscaled *copy* only.
- Only the alpha mask is upsampled (fast guided filter against the full-res
  frame, or RVM's built-in refiner).
- Compositing happens at full resolution; where alpha is 255 the output pixel
  is bit-identical to the camera pixel.

## Setup

Requires Windows 10/11 and **Python 3.12** (3.11 also works).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py
```

`download_models.py` fetches the models into `models/` (gitignored):

| File | Used for |
| --- | --- |
| `rvm_mobilenetv3_fp32.onnx`, `rvm_mobilenetv3_fp16.onnx` | Quality mode (Robust Video Matting) |
| `selfie_segmenter_landscape.tflite` | Performance mode (MediaPipe selfie segmenter) |
| `blaze_face_short_range.tflite` | Auto-framing (MediaPipe face detection) |

### Virtual camera driver (required for output)

Install one of these once:

- **OBS Virtual Camera** - install [OBS Studio](https://obsproject.com) (28+).
  The device appears as "OBS Virtual Camera". No alpha.
- **Unity Capture** - download <https://github.com/schellingb/UnityCapture>,
  run `Install\Install.bat` as administrator. The device appears as
  "Unity Video Capture" and carries **RGBA**, so Remove mode gives true
  transparency in OBS.

Without a driver the app still runs (preview, snapshots) and tells you the
driver is missing.

## Command line (Phase 1)

```powershell
.\.venv\Scripts\python.exe -m vcam --list-cameras
.\.venv\Scripts\python.exe -m vcam cli --mode blur --blur 60 --preview
.\.venv\Scripts\python.exe -m vcam cli --mode replace --bg C:\pics\office.jpg
.\.venv\Scripts\python.exe -m vcam cli --mode remove --rgba
.\.venv\Scripts\python.exe -m vcam --benchmark
.\.venv\Scripts\python.exe -m vcam cli --help
```
