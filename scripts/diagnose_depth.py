"""Short D435i IR/depth comparison; restores options and the live camera."""
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/depth_diagnostic"
BASE = "http://127.0.0.1:8766"


def api(path, data=None):
    request = urllib.request.Request(BASE + path, data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=20))


def coverage(array, scale):
    valid = array > 0
    values = array[valid].astype(float) * scale
    return {"valid_percent": round(100 * np.mean(valid), 3),
            "in_0_3_to_4m_percent": round(100 * np.mean(valid & (array * scale >= .3) & (array * scale <= 4)), 3),
            "percentiles_m": np.percentile(values, [1, 10, 50, 90, 99]).tolist() if len(values) else []}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = api("/api/status")
    if status["recording"]["state"] in {"countdown", "recording", "stopping"}:
        raise RuntimeError("An active recording must finish before camera diagnostics")
    report = {"started_unix": time.time(), "previous_camera": status["camera"], "cases": []}
    was_active = status["camera"]["active"]
    pipeline = None
    saved = {}
    advanced_json = None
    sensor = device = None
    try:
        api("/api/camera/stop", {})
        context = rs.context()
        device = next(d for d in context.query_devices() if d.get_info(rs.camera_info.serial_number) == status["camera"]["device"]["serial"])
        sensor = device.first_depth_sensor()
        for name in ("emitter_enabled", "laser_power", "enable_auto_exposure", "exposure", "gain", "visual_preset"):
            option = getattr(rs.option, name)
            if sensor.supports(option):
                saved[name] = sensor.get_option(option)
        try:
            advanced = rs.rs400_advanced_mode(device)
            if advanced.is_enabled():
                advanced_json = advanced.serialize_json()
                (OUT / "original_advanced_settings.json").write_text(advanced_json)
        except RuntimeError as error:
            report["advanced_settings_note"] = str(error)
        report["original_options"] = saved
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipeline = rs.pipeline(context)
        try:
            profile = pipeline.start(config)
            report["rgb_enabled"] = True
        except RuntimeError:
            pipeline = rs.pipeline(context)
            config.disable_stream(rs.stream.color)
            profile = pipeline.start(config)
            report["rgb_enabled"] = False
        sensor = profile.get_device().first_depth_sensor()
        scale = sensor.get_depth_scale()
        report["streams"] = [str(s) for s in profile.get_streams()]
        align = rs.align(rs.stream.color) if report["rgb_enabled"] else None
        cases = [("original", None), ("emitter_off", {"emitter_enabled": 0}),
                 ("laser_150", {"emitter_enabled": 1, "laser_power": 150}),
                 ("laser_300", {"emitter_enabled": 1, "laser_power": 300})]
        if advanced_json is not None and "visual_preset" in saved:
            for value in range(8):
                try:
                    description = sensor.get_option_value_description(rs.option.visual_preset, value)
                    if description and "medium" in description.lower() and "density" in description.lower():
                        cases.append(("medium_density", {"visual_preset": value, "emitter_enabled": 1, "laser_power": 150}))
                except RuntimeError:
                    pass
        for name, changes in cases:
            for key, value in (changes or {}).items():
                sensor.set_option(getattr(rs.option, key), value)
            for _ in range(30):
                pipeline.wait_for_frames(3000)
            readings = []
            for index in range(20):
                frames = pipeline.wait_for_frames(3000)
                depth = frames.get_depth_frame()
                raw = np.asanyarray(depth.get_data()).copy()
                readings.append(coverage(raw, scale))
                if index == 19:
                    cv2.imwrite(str(OUT / (name + "_depth16.png")), raw)
                    for stream_index in (1, 2):
                        ir = frames.get_infrared_frame(stream_index)
                        if ir:
                            cv2.imwrite(str(OUT / f"{name}_ir{stream_index}.png"), np.asanyarray(ir.get_data()))
                    if align is not None:
                        aligned = align.process(frames)
                        cv2.imwrite(str(OUT / (name + "_rgb.jpg")), np.asanyarray(aligned.get_color_frame().get_data()))
                        aligned_raw = np.asanyarray(aligned.get_depth_frame().get_data()).copy()
                        cv2.imwrite(str(OUT / (name + "_aligned_depth16.png")), aligned_raw)
                    sample = {"native": coverage(raw, scale),
                              "aligned": coverage(aligned_raw, scale) if align is not None else None}
                    for metadata_name in ("actual_exposure", "gain_level", "frame_counter", "frame_timestamp"):
                        metadata_key = getattr(rs.frame_metadata_value, metadata_name)
                        if depth.supports_frame_metadata(metadata_key):
                            sample[metadata_name] = depth.get_frame_metadata(metadata_key)
            sample.update(name=name, changes=changes, mean_valid_percent=float(np.mean([r["valid_percent"] for r in readings])),
                          mean_in_range_percent=float(np.mean([r["in_0_3_to_4m_percent"] for r in readings])))
            report["cases"].append(sample)
            print(json.dumps(sample), flush=True)
        report["ok"] = True
    except Exception as error:
        report.update(ok=False, error=str(error))
        raise
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        try:
            if sensor is not None:
                if "visual_preset" in saved:
                    sensor.set_option(rs.option.visual_preset, saved["visual_preset"])
                if advanced_json is not None:
                    rs.rs400_advanced_mode(device).load_json(advanced_json)
                for name in ("emitter_enabled", "laser_power", "exposure", "gain", "enable_auto_exposure"):
                    if name in saved:
                        sensor.set_option(getattr(rs.option, name), saved[name])
                report["restored_options"] = {name: sensor.get_option(getattr(rs.option, name)) for name in saved}
        except Exception as error:
            report["restore_options_error"] = str(error)
        finally:
            try:
                if was_active:
                    api("/api/camera/start", {})
            finally:
                report["finished_unix"] = time.time()
                (OUT / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
