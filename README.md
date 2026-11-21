# vcam-clone

Webcam background blur / removal / replacement for Windows that shows up as a
virtual camera in Zoom, Meet, Teams, Discord and OBS. No watermark, and the
person stays genuinely sharp at 1080p.

## The quality rule

The person's pixels are never downscaled.

- Capture is requested at native **1920x1080 MJPG** (USB 2.0 webcams silently
  fall back to 720p or lower in YUY2). The negotiated resolution, FPS and
  FOURCC are logged at startup, with a loud warning (log + banner above the
  preview) if they differ from what was requested.
- Segmentation runs on a downscaled *copy* only (480x270 for RVM at 1080p).
- Only the alpha mask is upsampled: a fast guided filter against the
  full-resolution frame, or RVM's built-in refiner.
- Compositing happens at full resolution. Where alpha is 255 the output pixel
  is bit-identical to the camera pixel.

## Quick start

Requires Windows 10/11 and **Python 3.12** (3.11 also works).

Double-click **`run.bat`**. On first use it creates `.venv`, installs the
pinned requirements, downloads the models and starts the app. If a PyInstaller
build exists (`dist\vcam-clone\vcam-clone.exe`) it starts that instead.

Manual setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe -m vcam
```

### Models

`scripts\download_models.py` fetches these into `models/` (gitignored):

| File | Used for | Source |
| --- | --- | --- |
| `rvm_mobilenetv3_fp32.onnx`, `rvm_mobilenetv3_fp16.onnx` | Quality mode (Robust Video Matting) | PeterL1n/RobustVideoMatting v1.0.0 release |
| `selfie_segmenter_landscape.tflite` | Performance mode (MediaPipe selfie segmenter) | Google MediaPipe model storage |
| `blaze_face_short_range.tflite` | Auto-framing (MediaPipe face detection) | Google MediaPipe model storage |

### Virtual camera driver (required for output)

Install one of these once. The app itself does not install drivers.

- **OBS Virtual Camera** - install [OBS Studio](https://obsproject.com) (28+).
  The device appears as "OBS Virtual Camera". No alpha channel.
- **Unity Capture** - download <https://github.com/schellingb/UnityCapture>
  and run `Install\Install.bat` as administrator. The device appears as
  "Unity Video Capture" and carries **RGBA**: with *Send alpha* enabled on
  the Output tab, Remove mode gives true transparency in OBS.

Without a driver the app still runs (preview, snapshots) and shows a banner
saying the driver is missing. After installing one, restart the app.

In Zoom / Meet / Teams / Discord pick "OBS Virtual Camera" (or "Unity Video
Capture") as the camera. Output is 1080p / 720p / 480p / 360p at 15 / 30 / 60
FPS.

## Using the app

| Tab | What is there |
| --- | --- |
| Background | Original, Blur (strength, bokeh), Remove (alpha + fallback colour), Replace (library grid, fit/fill, zoom, move, flip, blur, brightness, contrast, hue, saturation), Colour (solid / chroma green / blue) |
| Camera | Source picker with Refresh (virtual cameras are excluded), capture size / FPS / pixel format / API, software brightness-contrast-saturation-hue-gamma, hardware UVC zoom / pan / tilt / focus / exposure and the driver's own property dialog |
| Overlays | Logo / watermark layers (PNG, GIF, MP4...) with 9 anchors, size, offset, opacity, flip; name tag lower third |
| Enhance | Auto-framing (Zoom, Speed), Enhance Lighting, Pause Loop length, .cube LUT, presenter mode |
| Output | Virtual camera on/off, resolution, FPS, driver, RGBA |
| Settings | Model / processor, calibration, mirror preview, power saving, splash image, library folder, API keys, global hotkeys |

Under the preview: **Pause camera** (freeze frame), **Pause loop** (replays
the last few seconds forwards then backwards so there is no seam),
**Snapshot** (PNG in `Pictures\vcam-clone`) and **Mirror preview** (preview
only; the output is never mirrored).

**Closing the window (X) quits the app** and releases the webcam and the
virtual camera. Minimising hides it in the system tray, where the virtual
camera keeps running; the tray menu has the modes, next background, pause,
snapshot, sleep camera and Quit. Both behaviours can be changed under
Settings > Window (for example, make X keep the app in the tray). Only one
copy runs at a time: starting it again brings the running window forward.

### Background library

Drop files into the library (drag onto the grid, *Add files*, or copy them
into the folder; default `%APPDATA%\vcam-clone\backgrounds`). Supported: JPG,
PNG, WEBP, static and animated GIF, MP4 / WEBM / MOV (looped at the file's own
frame rate, muted).

*Search online* queries Pexels or Unsplash if you paste an API key in the
Settings tab. Keys are stored only in `%APPDATA%\vcam-clone\settings.json`,
never in the repository.

### Default hotkeys (global, editable in Settings)

| Action | Shortcut |
| --- | --- |
| Toggle blur | Ctrl+Alt+B |
| Toggle remove | Ctrl+Alt+R |
| Original | Ctrl+Alt+O |
| Next background | Ctrl+Alt+N |
| Pause / resume | Ctrl+Alt+P |
| Snapshot | Ctrl+Alt+S |

### Calibration

On first run (and from Settings > *Run calibration*) every model / processor
combination is benchmarked on your machine and the best-looking one that
holds the output frame rate is selected. Live FPS and inference time are shown
under the preview.

## Command line

```powershell
.\.venv\Scripts\python.exe -m vcam --list-cameras
.\.venv\Scripts\python.exe -m vcam --benchmark
.\.venv\Scripts\python.exe -m vcam cli --mode blur --blur 60 --preview
.\.venv\Scripts\python.exe -m vcam cli --mode replace --bg C:\pics\office.jpg
.\.venv\Scripts\python.exe -m vcam cli --mode remove --rgba
.\.venv\Scripts\python.exe -m vcam cli --mode blur --autoframe 1.5 --enhance 0.6 --name "Ada" --title "Engineer"
.\.venv\Scripts\python.exe -m vcam cli --help
```

`--no-vcam` runs without a virtual camera, `--snapshot out.png --seconds 5`
saves a frame and exits, `--input file.png|mp4` uses a file instead of a
camera (handy for testing).

## Building the exe

```powershell
.\build.bat
```

Produces `dist\vcam-clone\vcam-clone.exe` (PyInstaller one-folder build, models
copied next to the exe). `run.bat` then launches that build. The exe is a
windowed app; use `python -m vcam cli` for the command line.

## Performance

`scripts\profile_pipeline.py` times every stage on a fixed 1080p frame.
On the development machine (Radeon RX 9070 XT via DirectML, 16 threads):

| Stage (1920x1080) | Time |
| --- | --- |
| RVM fp32 model (DirectML) | ~5 ms |
| RVM + guided-filter mask upsample | ~9.5 ms |
| MediaPipe selfie + guided filter (CPU) | ~8 ms |
| Blend (replace / colour / remove) | ~1.8 ms |
| Blur background + blend | ~4.8 ms |
| Whole frame: replace / colour / remove | ~11.5 ms (85 FPS max) |
| Whole frame: blur | ~15 ms (67 FPS max) |

The profile showed compositing itself is under 2 ms with OpenCV's SIMD uint8
kernels, so it stays on the CPU: uploading 1080p frames to the GPU and
reading them back would cost more than it saves. The real costs were buffer
allocations in the mask refinement and refilling the solid colour plane; both
were fixed in the performance pass.

## Decisions and defaults

- **Python 3.12**, pinned versions in `requirements.txt`, everything in `.venv`.
- **Capture**: DirectShow by default (Media Foundation selectable), MJPG
  1920x1080 @ 30, capture on its own thread.
- **Quality mode default is RVM with the guided-filter upsample** rather than
  RVM's built-in refiner: the network sees a 480x270 copy
  (`downsample_ratio` 0.25 of 1080p) and keeps its recurrent state. It is
  about half the cost of the built-in refiner for a very similar edge.
  Calibration can still pick the built-in refiner when the GPU has headroom.
- **Performance mode** runs the MediaPipe selfie segmenter through the
  MediaPipe Tasks runtime (TFLite/XNNPACK on CPU, ~2-3 ms) rather than ONNX
  Runtime; converting the model to ONNX would add nothing.
- **Temporal EMA** is motion-adaptive (smooths static regions, follows moving
  edges) and runs on the small mask. *Edge tighten* contracts the soft edge
  slightly to remove the halo of old background around hair.
- **Blur** excludes the person from the blur kernel (normalised convolution),
  so there is no coloured halo around the outline.
- **Remove** sends RGBA only through Unity Capture; other drivers and apps
  get the fallback colour. In RGBA frames the transparent pixels also carry
  the fallback colour, so an app that ignores alpha never shows the real
  room; pixels with coverage keep the untouched camera colour, so OBS gets
  no colour fringe.
- **Tested output path**: Unity Capture at 1920x1080 (BGR and RGBA), read
  back through a DirectShow graph: same resolution and sharpness as the
  composited frame, not mirrored. OpenCV's own DirectShow reader cannot open
  the Unity Capture filter, which is a limitation of that reader only.
- **Auto-framing** crops from the full-resolution frame with a critically
  damped spring. Zooming in on a 1080p camera with a 1080p output must
  upscale; the app warns above the preview. *Never upscale* caps the zoom so
  the crop never has fewer pixels than the output (use a 720p output to zoom
  up to 1.5x pixel-exact).
- **Enhance Lighting** estimates gamma, white balance and a CLAHE-based local
  contrast map on a small copy, smooths them over time and applies them to
  the full-resolution frame.
- **Pause Loop** keeps the last N seconds as JPEG (quality 93) on a worker
  thread; turn the buffer off in Enhance to save a little CPU.
- **Power saving**: the webcam is released when the window is hidden in the
  tray and the virtual camera is off or has no driver, or after the sleep
  timer (minutes in the tray, 0 = never). The app cannot see whether another
  program is currently reading the virtual camera, so "output idle" means
  "output switched off". While asleep the virtual camera shows the splash
  image; showing the window wakes the camera.
- **Hardware UVC controls** go through OpenCV `CAP_PROP_*`. OpenCV cannot
  query each control's range, so the sliders use generic ranges; the *Driver
  dialog* button opens the camera's own property page with exact ranges.
- **Virtual cameras are excluded** from the source list by name (OBS, Unity
  Capture, XSplit VCam, Snap Camera, NDI, DroidCam and similar).
- **Settings** live in `%APPDATA%\vcam-clone\settings.json`, the log in
  `%APPDATA%\vcam-clone\vcam.log`.
- **Extras**: `.cube` 3D LUTs (Enhance tab) and presenter mode (live screen
  capture behind the matted person).

## Project layout

```
vcam/
  capture.py       threaded capture, MJPG 1080p negotiation + warnings
  devices.py       camera enumeration, virtual camera filter
  segmentation.py  RVM / MediaPipe, EMA, guided-filter alpha upsample
  compositor.py    blend, blur / bokeh, background layer, colour adjust
  media.py         image / GIF / video sources
  overlays.py      logo layers, name tag
  autoframe.py     face detection + spring crop window
  lighting.py      enhance lighting
  pause.py         freeze + ping-pong loop buffer
  output.py        pyvirtualcam (OBS / Unity Capture)
  engine.py        the pipeline thread
  calibrate.py     model / provider benchmark
  hotkeys.py, splash.py, stock.py, lut.py, presenter.py, settings.py
  cli.py           command-line front end
  ui/              PySide6 app (main window, tabs, library, tray)
scripts/
  download_models.py, profile_pipeline.py
```
