
from vidgear.gears import CamGear
import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import time

_FONT_PATH = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
try:
    _font_title = ImageFont.truetype(_FONT_PATH, 15)
    _font_body = ImageFont.truetype(_FONT_PATH, 13)
except OSError:
    _font_title = _font_body = ImageFont.load_default()


def main():

    def video_manifest_extractor(source):
        """
        Function to extract metadata from a YouTube video source
          and find the desired resolution URL.

        Parameters:
        source (str): Video source URL (ex. "<https://youtu.be/bvetuLwJIkA>")

        Returns:
        str: Desired resolution video URL
        """
        stream = CamGear(source=source, stream_mode=True, logging=True,
                         time_delay=0).start()
        video_metadata = stream.ytv_metadata

        print(video_metadata.keys())
        print(video_metadata["fps"])
        print(video_metadata["format"])
        print(video_metadata["format_index"])

        resolutions = [format["resolution"] for format in video_metadata["formats"]]
        for res in resolutions:
            print(res)

        resolution_desiree = '1280x720'
        for format in video_metadata["formats"]:
            if format["resolution"] == resolution_desiree:
                VIDEO = format["url"]
                return VIDEO

    source = "https://youtu.be/z545k7Tcb5o"
    VIDEO = video_manifest_extractor(source)
    print(VIDEO)

    # Load OpenVINO model for better performance
    MODEL = "models/yolov8s.pt"
    # MODEL = "models/yolo11s.pt"
    # MODEL = "models/yolo26s.pt"
    model = YOLO(MODEL)

    # Get class names dictionary
    CLASS_NAMES_DICT = model.model.names
    print(CLASS_NAMES_DICT)

    # model_openvino = YOLO("models/yolov8s_openvino_model/", task="detect")
    # model_openvino = YOLO("models/yolo11s_openvino_model", task='detect')
    model_openvino = YOLO("models/yolo26s_openvino_model", task='detect')
    colors = sv.ColorPalette.LEGACY

    video_info = sv.VideoInfo.from_video_path(VIDEO)

    # Calculate the scaling coefficient based on the video width and
    #  the desired output width (1280)
    coef = video_info.width / 1280
    # print(coef)

    # polygon design
    #  ----> x
    # |         (x4,y4)   (x3,y3)
    # |              +-------+
    #               +-------+
    # y            +-------+
    #         (x1,y1)    (x2,y2)

    # 3 polygons so 3 values in each coordinate from left to right 
    #    [zone1,zone2, zone3]
    x1 = [-160, -25, 971]
    y1 = [405, 710, 671]
    x2 = [112, 568, 1480]
    y2 = [503, 710, 671]
    x3 = [557, 706, 874]
    y3 = [195, 212, 212]
    x4 = [411, 569, 749]
    y4 = [195, 212, 212]

    # Scale coordinates according to the video flow and the 
    # aspect ratio of the displayed video
    x1, y1, x2, y2, x3, y3, x4, y4 = map(
        lambda vals: list(map(lambda val: val * coef, vals)),
        [x1, y1, x2, y2, x3, y3, x4, y4]
    )

    # Find the centroid or third point from top of the polygon
    # e.g.: ((x1 + 2* x4) / 3) for drawing counting lines
    x14 = list(map(lambda x1, x4: (x1 + 2 * x4) / 3, x1, x4))
    y14 = list(map(lambda y1, y4: (y1 + 2 * y4) / 3, y1, y4))
    x23 = list(map(lambda x2, x3: (x2 + 2 * x3) / 3, x2, x3))
    y23 = list(map(lambda y2, y3: (y2 + 2 * y3) / 3, y2, y3))

    # Polygon zones defined from left to right (make sure in the same order
    #  as the linezone)
    polygons = [
        np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        for x1, y1, x2, y2, x3, y3, x4, y4
        in zip(x1, y1, x2, y2, x3, y3, x4, y4)
    ]
    # initialize our zones
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            # frame_resolution_wh=video_info.resolution_wh
        )
        for polygon
        in polygons
    ]
    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=2,
            text_thickness=1,
            text_scale=0.5,
        )
        for index, zone
        in enumerate(zones)
    ]
    label_annotators = [
        sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            color=colors.by_idx(index),
            text_thickness=1,
            text_scale=0.5,
        )
        for index
        in range(len(zones))
    ]
    box_annotators = [
        sv.BoxAnnotator(
            color=colors.by_idx(index),
            thickness=1,
            # text_thickness=1,
            # text_scale=0.5
            )
        for index
        in range(len(polygons))
    ]
    # box_annotators = [
    #     sv.BoundingBoxAnnotator(
    #         color=colors.by_idx(index),
    #         thickness=1,
    #         )
    #     for index
    #     in range(len(polygons))
    # ]

    trace_annotators = [
        sv.TraceAnnotator(
            color=colors.by_idx(index),
            thickness=1,
            trace_length=video_info.fps * 1.5,
            position=sv.Position.BOTTOM_CENTER,
            )
        for index
        in range(len(polygons))
    ]
    lines_start = [
        sv.Point(x14, y14)
        for x14, y14
        in zip(x14, y14)
    ]
    lines_end = [
        sv.Point(x23, y23)
        for x23, y23
        in zip(x23, y23)
    ]
    positions = [
        (sv.Position.CENTER, sv.Position.CENTER),
        (sv.Position.CENTER, sv.Position.CENTER),
        (sv.Position.CENTER, sv.Position.CENTER),
    ]
    line_zones = [
        sv.LineZone(start=line_start, end=line_end,
                    triggering_anchors=position)
        for line_start, line_end, position
        in zip(lines_start, lines_end, positions)
    ]
    # for automatic line zone annotator not use here want to use a custom one
    line_zone_annotators = [
        sv.LineZoneAnnotator(thickness=1,
                             color=colors.by_idx(index),
                             text_thickness=1,
                             text_scale=0.5,
                             text_offset=4)
        for index
        in range(len(line_zones))
    ]
    # couting line zone text position
    # text_pos = [
    #     sv.Point(x=100, y=320),
    #     sv.Point(x=700, y=320),
    #     sv.Point(x=1077, y=320)
    # ]
    # initialyze ByteTracker
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=100,
        minimum_matching_threshold=0.8,
        frame_rate=video_info.fps
    )
    # byte_tracker = sv.ByteTrack()
    fps_monitor = sv.FPSMonitor()
    # heat_map = sv.HeatMapAnnotator()
    # smoother = sv.DetectionsSmoother()
    # intialize the source coordinate for speed estimation
    SOURCES = np.array([[
        [x4[0], y4[0]],
        [x3[0], y3[0]],
        [x2[0], y2[0]],
        [x1[0], y1[0]]

    ], [[x4[1], y4[1]],
        [x3[1], y3[1]],
        [x2[1], y2[1]],
        [x1[1], y1[1]]],
          [[x4[2], y4[2]],
           [x3[2], y3[2]],
           [x2[2], y2[2]],
           [x1[2], y1[2]]
           ]
           ])
    # initialize Target real(in meters) coordinate 
    # zone1 in meters
    TARGET_WIDTH = 6
    TARGET_HEIGHT = 75
    TARGETS = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ])
    # zone 2 in meters
    TARGET_WIDTH = 6
    TARGET_HEIGHT = 85
    TARGETS = np.append(TARGETS, np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]), axis=0)
    # zone3 in meters
    TARGET_WIDTH = 6
    TARGET_HEIGHT = 80
    TARGETS = np.append(TARGETS, np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]), axis=0)

    TARGETS = TARGETS.reshape(3, 4, 2)
    # class searching transformation matrix between
    #  SOURCE and TARGET to get speed estimation

    class ViewTransformer:

        def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
            source = source.astype(np.float32)
            target = target.astype(np.float32)
            self.m = cv2.getPerspectiveTransform(source, target)

        def transform_points(self, points: np.ndarray) -> np.ndarray:
            if points.size == 0:
                return points

            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(
                    reshaped_points, self.m)
            return transformed_points.reshape(-1, 2)

    # create the transformers matrix for each zone
    view_transformers = [
        ViewTransformer(source=s, target=t)
        for s, t
        in zip(SOURCES, TARGETS)
    ]
    # car, motorcycle, bus, truck from coco classes
    selected_classes = [2, 3, 5, 7] 
    # initialize the dictionary that we 
    # will use to store the coordinates for each zone
    coordinates = defaultdict(lambda: deque(maxlen=30))
    coordinates = np.append(coordinates, defaultdict(lambda: deque(maxlen=30)))
    coordinates = np.append(coordinates, defaultdict(lambda: deque(maxlen=30)))
    # Timestamps réels (live stream) — parallèles aux coordonnées
    timestamps = defaultdict(lambda: deque(maxlen=30))
    timestamps = np.append(timestamps, defaultdict(lambda: deque(maxlen=30)))
    timestamps = np.append(timestamps, defaultdict(lambda: deque(maxlen=30)))

    # Speed limits per zone (km/h)
    SPEED_LIMITS = [110, 110, 70]
    # Cumulative count of speeding vehicles per zone
    speeding_counts = [0, 0, 0]
    # Track already-counted tracker IDs per zone to avoid double-counting
    speeding_tracker_ids = [set(), set(), set()]
    # Max speed observed per zone and overall
    max_speeds = [0.0, 0.0, 0.0]
    max_speed_global = [0.0]  # list to allow mutation inside nested function
    # Speed samples history per zone and per tracker_id — used to compute peak speed
    speed_samples = [defaultdict(list) for _ in range(3)]

    # Cache: stores (key, panel_bgr, panel_alpha_f) to avoid re-rendering every frame
    _panel_cache = [None, None, None]

    def _render_panel(speed_limits: list, counts: list,
                      zone_max_speeds: list, global_max: float):
        """Render the stats panel once as (bgr_array, alpha_float_array) via PIL.

        Only called when stats actually change; result is cached.
        """
        panel_w, line_h, padding = 295, 24, 10
        n_lines = 1 + len(speed_limits) + 1 + 1
        panel_h = padding * 2 + n_lines * line_h + (n_lines - 1) * 2

        pil_panel = Image.new("RGBA", (panel_w, panel_h), (10, 10, 10, 145))
        draw = ImageDraw.Draw(pil_panel)

        draw.text((padding, padding), "Dépassements vitesse",
                  font=_font_title, fill=(255, 255, 255, 255))

        for idx, (limit, count, zmax) in enumerate(
            zip(speed_limits, counts, zone_max_speeds)
        ):
            c = colors.by_idx(idx)
            y = padding + (idx + 1) * (line_h + 2)
            text = f"Zone {idx + 1} (>{limit} km/h) : {count} veh.  max {int(zmax)} km/h"
            draw.text((padding, y), text, font=_font_body, fill=(c.r, c.g, c.b, 255))

        sep_y = padding + (len(speed_limits) + 1) * (line_h + 2)
        draw.line([(padding, sep_y), (panel_w - padding, sep_y)],
                  fill=(180, 180, 180, 200), width=1)
        draw.text((padding, sep_y + 5), f"Max globale : {int(global_max)} km/h",
                  font=_font_title, fill=(0, 215, 255, 255))

        arr = np.array(pil_panel)                           # H×W×4  uint8 RGBA
        panel_rgb = arr[:, :, :3][:, :, ::-1].copy()       # RGB → BGR
        panel_alpha = (arr[:, :, 3:] / 255.0).astype(np.float32)  # H×W×1 [0,1]
        return panel_rgb, panel_alpha, panel_w, panel_h

    def draw_speed_stats(
        frame: np.ndarray,
        speed_limits: list,
        counts: list,
        zone_max_speeds: list,
        global_max: float,
    ) -> np.ndarray:
        """Composite the cached stats panel onto the frame using numpy alpha-blending.

        PIL is only called when the stats data actually changes (rare), making
        per-frame cost a simple numpy slice operation instead of two full-frame
        colour conversions.
        """
        key = (tuple(counts), tuple(int(z) for z in zone_max_speeds), int(global_max))
        if _panel_cache[0] != key:
            panel_bgr, panel_alpha, pw, ph = _render_panel(
                speed_limits, counts, zone_max_speeds, global_max)
            _panel_cache[0] = key
            _panel_cache[1] = panel_bgr
            _panel_cache[2] = (panel_alpha, pw, ph)

        panel_bgr = _panel_cache[1]
        panel_alpha, pw, ph = _panel_cache[2]

        h, w = frame.shape[:2]
        x0, y0 = w - pw - 10, 10
        roi = frame[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
        frame[y0:y0 + ph, x0:x0 + pw] = (
            roi * (1.0 - panel_alpha) + panel_bgr.astype(np.float32) * panel_alpha
        ).astype(np.uint8)
        return frame

    # frame processing

    def process_frame(frame: np.ndarray) -> np.ndarray:
        speed_labels = [], [], []
        # Force CPU device: OpenVINO on Intel iGPU triggers an IGC kernel error
        # (intersecting register V37/V38) causing NaN outputs and garbage detections
        results = model_openvino(frame, imgsz=640, verbose=False)[0]
        # results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, selected_classes)] # filer on selected classes
        detections = byte_tracker.update_with_detections(detections)
        # detections = smoother.update_with_detections(detections)

        # copy frame before annotate                     
        annotated_frame = frame.copy()

        for i, (zone,
                zone_annotator,
                box_annotator,
                trace_annotator,
                line_zone,
                line_zone_annotator,
                label_annotator,
                line_start,
                line_end,
                view_transformer,
                speed_label,
                    coordinate,
                    timestamp) in enumerate(zip(
                    zones,
                    zone_annotators,
                    box_annotators,
                    trace_annotators,
                    line_zones,
                    line_zone_annotators,
                    label_annotators,
                    lines_start,
                    lines_end,
                    view_transformers,
                    speed_labels,
                    coordinates,
                    timestamps)):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            points = detections_filtered.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER)
            # Intégrer le transformateur de vue dans un pipeline de détection existant
            points = view_transformer.transform_points(points=points).astype(int)
            for tracker_id, [_, y] in zip(detections_filtered.tracker_id, points):
                coordinate[tracker_id].append(y)
                timestamp[tracker_id].append(time.time())

            # wait to have enough data (au moins 2 points et 0.3 s écoulées)
            for tracker_id in detections_filtered.tracker_id:
                if len(timestamp[tracker_id]) < 2:
                    speed_label.append(f"#{tracker_id}")
                else:
                    try:
                        elapsed = timestamp[tracker_id][-1] - timestamp[tracker_id][0]
                        if elapsed <= 0:
                            speed_label.append(f"#{tracker_id}")
                            continue
                        coordinate_start = coordinate[tracker_id][-1]
                        coordinate_end = coordinate[tracker_id][0]
                        distance = abs(coordinate_start - coordinate_end)
                        speed = distance / elapsed * 3.6
                        # Accumulate speed samples and derive peak speed for this vehicle
                        speed_samples[i][tracker_id].append(speed)
                        peak_speed = max(speed_samples[i][tracker_id])
                        speed_label.append(f"{int(peak_speed)} km/h")
                        # Update max speeds using peak speed
                        if peak_speed > max_speeds[i]:
                            max_speeds[i] = peak_speed
                        if peak_speed > max_speed_global[0]:
                            max_speed_global[0] = peak_speed
                        # Count speeding vehicles once, based on peak speed
                        if peak_speed > SPEED_LIMITS[i] and tracker_id not in speeding_tracker_ids[i]:
                            speeding_tracker_ids[i].add(tracker_id)
                            speeding_counts[i] += 1

                    except Exception as e:
                        speed_label.append(f"#{tracker_id}")
                        print(f"An error occurred: {e}")
            # labels = [
            # f"#{tracker_id} "
            # for _,_,_,_,tracker_id in detections_filtered]
            line_zone.trigger(detections=detections_filtered)

            annotated_frame = sv.draw_line(scene=annotated_frame,
                                           start=line_start,
                                           end=line_end,
                                           color=colors.by_idx(i))
            # annotated_frame = zone_annotator.annotate(scene=annotated_frame,
            #  label=f"Dir. Ouest : {i+random.randint(0,100)}")
            direction_label = "Dir. West" if i == 0 else "Dir. East"
            total_count = line_zone.in_count + line_zone.out_count
            annotated_frame = zone_annotator.annotate(
                scene=annotated_frame,
                label=f"{direction_label} : {total_count}")

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections_filtered,
                labels=speed_label)

            # annotated_frame=line_zone_annotator.annotate(
            # annotated_frame,line_counter=line_zone )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections_filtered)

            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections_filtered)
            # print(f"Line Zone In Count: {line_zone.in_count}", {i: direction_label})
            # print(f"Line Zone Out Count: {line_zone.out_count}", {i: direction_label})
        draw_speed_stats(annotated_frame, SPEED_LIMITS, speeding_counts, max_speeds, max_speed_global[0])
        return annotated_frame
    # for direct show
    cap = cv2.VideoCapture(VIDEO)
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = video_info.fps or cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"FPS: {fps}")
    print(f"image : {width}x{height}")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # frame=cv2.resize(frame,(1280,720))
        show = process_frame(frame)
        fps_monitor.tick()
        current_fps = fps_monitor.fps
        fps_text = f"FPS: {current_fps:.0f}"
        cv2.putText(show, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Counting - Speed Estimation", show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Limiter les FPS à 25
        time.sleep(max(1/25 - (time.time() - start_time), 0))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
