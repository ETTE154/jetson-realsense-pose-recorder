# Agent instructions

Read README.md before installation and docs/dataset_csv.md before changing data
semantics. Follow the numbered setup and acceptance sequence. Commands run from
the repository root on the Jetson unless marked as PC commands.

## Preserve the working environment

- Inspect platform, JetPack/CUDA/TensorRT/Python, disk space and USB speed first.
- Check GET http://127.0.0.1:8766/api/status before touching a running server.
  Do not interrupt countdown, recording or stopping sessions, kill unrelated
  processes, or overwrite user changes.
- Keep system JetPack and .venv separate from MediaPipe, EgoForce GPU and export
  environments. Do not upgrade system NumPy/OpenCV, replace Jetson GPU PyTorch
  with CPU PyTorch, reflash or patch the kernel during ordinary setup.
- Build TensorRT engines on the target. Keep downloaded models, environments,
  real images, sensor data, passwords and personal host details out of Git.
  Respect pinned upstream revisions/checksums and licenses.
- Bind the server to 127.0.0.1 and connect through SSH forwarding. Obtain missing
  connection details from the user as needed. Do not store passwords in files or
  command arguments, or bypass SSH host-key checks. USB permissions are scoped
  to D435i 8086:0b3a, video group, mode 0660.

## Preserve recording and dataset behavior

- IMU movie rendering is intentionally disabled in v1.4.2. Preserve native IMU
  CSV and live UI plots, six MP4 outputs, and the 1280x960 four-view combined
  movie. Do not reactivate imu.mp4 or append an IMU panel to satisfy an old test.
  imu_video.py is not part of the active recording path.
- Preserve the server-side 5-second capture-time gate, stop cutoff, original
  source frame identity, clock domains, capture generation and blank missing
  values. Never draw old pose on new RGB or fabricate confidence/depth/IMU data.
- On the reference device, camera is ~30Hz, video ~15Hz, accel ~63Hz and gyro
  ~200Hz. CSV preserves native events independently of video cadence.
- Direct SDK camera checks need the server stopped; HTTP validation scripts
  need the server running. Read the script before invoking it.

## Verify and report

- Run .venv/bin/python -m unittest discover -s tests -v and, if Node 18+ is
  present, node tests/test_imu_ui.js. SDK tests need the RSUSB environment from
  README section 4. Unit tests do not prove actual camera operation.
- Full setup acceptance includes actual RGB/Depth/IMU reception, model changes,
  a visible hand/person, dataset.csv and a 90-second continuity run. Check
  existing recording state before scripts that record or change models.
- verify_dataset.py exit 2 is partial verification (no detected joints), not
  complete success. Do not turn missing hardware into simulated success.
- Report the tested commit, device versions, checks, errors/dropped counts,
  report paths and unverified items. Historical results in docs/validation.md
  are not measurements for a freshly installed machine.
