"""Synthetic recording codec probe; never opens the RealSense camera."""
from pathlib import Path
import json
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1] / "validation" / "recording_codec_probe"
root.mkdir(parents=True, exist_ok=True)
results = []
for name, suffix in [("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi"), ("VP80", ".webm")]:
    path = root / (name + suffix)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*name), 15.0, (640, 480))
    opened = writer.isOpened()
    if opened:
        for index in range(15):
            image = np.zeros((480,640,3),dtype=np.uint8)
            image[:]=(index*9,60,160)
            cv2.putText(image,f'{name} frame {index}',(60,240),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            writer.write(image)
    writer.release()
    decoded = 0
    cap = cv2.VideoCapture(str(path)) if path.exists() else None
    if cap is not None:
        while True:
            ok,image = cap.read()
            if not ok: break
            decoded += 1
        cap.release()
    results.append({"fourcc":name,"path":str(path),"opened":opened,"decoded_frames":decoded,
                    "bytes":path.stat().st_size if path.exists() else 0,"fps":15,"width":640,"height":480})
report={"opencv":cv2.__version__,"codecs":results}
(root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
