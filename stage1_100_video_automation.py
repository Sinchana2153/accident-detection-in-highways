# ================================================================
# STAGE 1 - 100 VIDEO AUTOMATION
# ================================================================
#
# FEATURES
# ------------------------------------------------
# 1. Automatically finds all videos in dataset folder
# 2. Automatically matches ground-truth names with video names
#    even when extensions are different/missing
#
#    video1
#    video1.avi
#    video1.mp4
#    video1.mkv
#
#    are treated as the same video.
#
# 3. Analysis target = 15 FPS
# 4. YOLO input = 640 x 640
# 5. YOLO11n
# 6. ByteTrack
# 7. Vehicle tracking
# 8. Trajectory generation
# 9. Collision detection
# 10. Accident / Non-Accident prediction
# 11. Ground truth comparison only for Accident/Non-Accident
# 12. TP / TN / FP / FN
# 13. Accuracy / Precision / Recall / F1
# 14. Excel output
#
# Ground truth:
# D:\accidentproject\ground_truth.xlsx
#
# Expected columns:
# video name
# collision
# timestamp
#
# IMPORTANT:
# Ground-truth timestamp is NOT compared with prediction timestamp.
# Trajectory is NOT used for TP/TN/FP/FN.
#
# ================================================================


from pathlib import Path
import math
import traceback

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_DIR = Path(r"D:\accidentproject")

DATASET_DIR = PROJECT_DIR / "dataset"

GROUND_TRUTH_FILE = PROJECT_DIR / "ground_truth.xlsx"

MODEL_FILE = PROJECT_DIR / "yolo11n.pt"

OUTPUT_DIR = PROJECT_DIR / "stage1_results"

OUTPUT_EXCEL = OUTPUT_DIR / "stage1_results.xlsx"

TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"


# Supported video formats
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
    ".mpeg",
    ".mpg"
}


# ================================================================
# ANALYSIS SETTINGS
# ================================================================

TARGET_FPS = 15.0

YOLO_WIDTH = 640
YOLO_HEIGHT = 640

CONFIDENCE = 0.25

TRACKER = "bytetrack.yaml"


# COCO vehicle classes:
#
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = [2, 3, 5, 7]


# Collision detection settings

COLLISION_DISTANCE_THRESHOLD = 50.0

MIN_CLOSE_FRAMES = 3

APPROACH_MARGIN = 1.0


# ================================================================
# NORMALIZE VIDEO NAME
# ================================================================

def normalize_video_name(name):
    """
    Converts a video name into a common matching key.

    Examples:

        video1
        video1.avi
        video1.mp4
        VIDEO1.AVI

    all become:

        video1

    This allows automatic ground-truth matching.
    """

    if pd.isna(name):
        return ""

    text = str(name).strip()

    if not text:
        return ""

    # Remove folder path if present
    text = Path(text).name

    # Remove extension
    text = Path(text).stem

    # Normalize case
    text = text.lower().strip()

    return text


# ================================================================
# NORMALIZE GROUND-TRUTH LABEL
# ================================================================

def normalize_label(value):

    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    accident_values = {
        "ACCIDENT",
        "YES",
        "Y",
        "TRUE",
        "1",
        "COLLISION",
        "CRASH",
        "ACCIDENT HAPPENED"
    }

    non_accident_values = {
        "NON_ACCIDENT",
        "NON-ACCIDENT",
        "NO_ACCIDENT",
        "NO-ACCIDENT",
        "NO ACCIDENT",
        "NO",
        "N",
        "FALSE",
        "0",
        "NORMAL",
        "NONACCIDENT",
        "NO COLLISION",
        "TRAFFIC"
    }

    if text in accident_values:
        return "ACCIDENT"

    if text in non_accident_values:
        return "NON_ACCIDENT"

    if "NON" in text and (
        "ACCIDENT" in text
        or "COLLISION" in text
    ):
        return "NON_ACCIDENT"

    if "ACCIDENT" in text:
        return "ACCIDENT"

    if "COLLISION" in text:
        return "ACCIDENT"

    if "CRASH" in text:
        return "ACCIDENT"

    if "NORMAL" in text:
        return "NON_ACCIDENT"

    if "TRAFFIC" in text:
        return "NON_ACCIDENT"

    return text


# ================================================================
# FORMAT TIMESTAMP
# ================================================================

def format_timestamp(seconds):

    if seconds is None:
        return ""

    try:
        seconds = float(seconds)
    except Exception:
        return ""

    if not math.isfinite(seconds):
        return ""

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:04.1f}"
    )


# ================================================================
# SAFE FLOAT
# ================================================================

def safe_float(
    value,
    default=0.0
):

    try:

        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except Exception:

        return default


# ================================================================
# LOAD GROUND TRUTH
# ================================================================

def load_ground_truth(file_path):

    if not file_path.exists():

        raise FileNotFoundError(
            f"Ground truth file not found:\n"
            f"{file_path}"
        )

    print()
    print(
        "Loading ground truth..."
    )

    df = pd.read_excel(
        file_path
    )

    print(
        "Ground truth columns:",
        list(df.columns)
    )

    if df.empty:

        raise ValueError(
            "Ground truth Excel file is empty."
        )

    # ------------------------------------------------------------
    # Find columns
    # ------------------------------------------------------------

    video_col = None
    label_col = None
    timestamp_col = None

    for col in df.columns:

        normalized = str(
            col
        ).strip().lower()

        if normalized == "video name":
            video_col = col

        elif normalized == "collision":
            label_col = col

        elif normalized == "timestamp":
            timestamp_col = col

    # ------------------------------------------------------------
    # Fallback search
    # ------------------------------------------------------------

    if video_col is None:

        for col in df.columns:

            normalized = str(
                col
            ).strip().lower()

            if (
                "video" in normalized
                or "file" in normalized
                or "filename" in normalized
            ):

                video_col = col
                break

    if label_col is None:

        for col in df.columns:

            normalized = str(
                col
            ).strip().lower()

            if (
                "collision" in normalized
                or "accident" in normalized
                or "label" in normalized
            ):

                label_col = col
                break

    if timestamp_col is None:

        for col in df.columns:

            normalized = str(
                col
            ).strip().lower()

            if (
                "timestamp" in normalized
                or normalized == "time"
            ):

                timestamp_col = col
                break

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    if video_col is None:

        raise ValueError(
            "Could not find 'video name' column."
        )

    if label_col is None:

        raise ValueError(
            "Could not find 'collision' column."
        )

    print(
        "Ground truth video column :",
        video_col
    )

    print(
        "Ground truth label column :",
        label_col
    )

    print(
        "Ground truth timestamp    :",
        timestamp_col
    )

    # ------------------------------------------------------------
    # Create mapping
    # ------------------------------------------------------------

    gt_mapping = {}

    for _, row in df.iterrows():

        original_name = row[
            video_col
        ]

        key = normalize_video_name(
            original_name
        )

        if not key:
            continue

        label = normalize_label(
            row[label_col]
        )

        timestamp = ""

        if timestamp_col is not None:

            value = row[
                timestamp_col
            ]

            if not pd.isna(value):

                timestamp = str(
                    value
                )

        gt_mapping[key] = {
            "original_name":
                str(original_name),

            "label":
                label,

            "timestamp":
                timestamp
        }

    print(
        "Ground truth rows loaded  :",
        len(gt_mapping)
    )

    return (
        gt_mapping,
        video_col,
        label_col,
        timestamp_col
    )


# ================================================================
# FIND VIDEOS
# ================================================================

def find_videos(
    dataset_dir
):

    if not dataset_dir.exists():

        raise FileNotFoundError(
            f"Dataset folder not found:\n"
            f"{dataset_dir}"
        )

    videos = []

    for path in dataset_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in VIDEO_EXTENSIONS:

            videos.append(path)

    videos.sort(
        key=lambda p: p.name.lower()
    )

    return videos


# ================================================================
# GET VIDEO INFORMATION
# ================================================================

def get_video_info(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    original_fps = safe_float(
        cap.get(
            cv2.CAP_PROP_FPS
        ),
        0
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = 0.0

    if original_fps > 0:

        duration = (
            frame_count
            /
            original_fps
        )

    cap.release()

    return {
        "original_fps":
            original_fps,

        "frame_count":
            frame_count,

        "width":
            width,

        "height":
            height,

        "duration":
            duration
    }


# ================================================================
# FRAME SAMPLING
# ================================================================

def should_process_frame(
    frame_index,
    original_fps,
    target_fps
):

    if original_fps <= 0:

        return True

    # If original video is already <= 15 FPS,
    # process every available frame.

    if original_fps <= target_fps:

        return True

    interval = (
        original_fps
        /
        target_fps
    )

    if interval <= 1:

        return True

    # Use a floating-point accumulator approach
    # approximately through modulo selection.

    return (
        frame_index
        %
        max(
            1,
            round(interval)
        )
        == 0
    )


# ================================================================
# DISTANCE
# ================================================================

def center_distance(
    point1,
    point2
):

    dx = (
        point1[0]
        -
        point2[0]
    )

    dy = (
        point1[1]
        -
        point2[1]
    )

    return math.sqrt(
        dx * dx
        +
        dy * dy
    )


# ================================================================
# COLLISION ANALYSIS
# ================================================================

def analyze_collision(
    vehicle_history
):

    if len(vehicle_history) < 2:

        return {
            "collision_detected":
                "NO",

            "collision_frame":
                "",

            "collision_timestamp":
                "",

            "collision_distance":
                "",

            "vehicle_1":
                "",

            "vehicle_2":
                "",

            "close_frame_count":
                0
        }

    vehicle_ids = list(
        vehicle_history.keys()
    )

    best_candidate = None

    # ------------------------------------------------------------
    # Compare vehicle pairs
    # ------------------------------------------------------------

    for i in range(
        len(vehicle_ids)
    ):

        id1 = vehicle_ids[i]

        trajectory1 = (
            vehicle_history[id1]
        )

        if len(trajectory1) < 2:
            continue

        lookup1 = {}

        for point in trajectory1:

            lookup1[
                point["frame"]
            ] = point

        for j in range(
            i + 1,
            len(vehicle_ids)
        ):

            id2 = vehicle_ids[j]

            trajectory2 = (
                vehicle_history[id2]
            )

            if len(trajectory2) < 2:
                continue

            lookup2 = {}

            for point in trajectory2:

                lookup2[
                    point["frame"]
                ] = point

            common_frames = sorted(
                set(
                    lookup1.keys()
                )
                &
                set(
                    lookup2.keys()
                )
            )

            if not common_frames:
                continue

            close_frames = []

            for frame in common_frames:

                p1 = lookup1[
                    frame
                ]

                p2 = lookup2[
                    frame
                ]

                distance = center_distance(
                    (
                        p1["x"],
                        p1["y"]
                    ),
                    (
                        p2["x"],
                        p2["y"]
                    )
                )

                if (
                    distance
                    <=
                    COLLISION_DISTANCE_THRESHOLD
                ):

                    close_frames.append(
                        (
                            frame,
                            distance
                        )
                    )

            if (
                len(close_frames)
                <
                MIN_CLOSE_FRAMES
            ):

                continue

            # ----------------------------------------------------
            # Minimum distance
            # ----------------------------------------------------

            minimum = min(
                close_frames,
                key=lambda x: x[1]
            )

            candidate = {

                "vehicle_1":
                    id1,

                "vehicle_2":
                    id2,

                "frame":
                    minimum[0],

                "distance":
                    minimum[1],

                "close_count":
                    len(close_frames)
            }

            if best_candidate is None:

                best_candidate = candidate

            else:

                if (
                    candidate["distance"]
                    <
                    best_candidate[
                        "distance"
                    ]
                ):

                    best_candidate = candidate

    # ------------------------------------------------------------
    # No collision
    # ------------------------------------------------------------

    if best_candidate is None:

        return {
            "collision_detected":
                "NO",

            "collision_frame":
                "",

            "collision_timestamp":
                "",

            "collision_distance":
                "",

            "vehicle_1":
                "",

            "vehicle_2":
                "",

            "close_frame_count":
                0
        }

    # ------------------------------------------------------------
    # Collision detected
    # ------------------------------------------------------------

    collision_frame = (
        best_candidate["frame"]
    )

    # This is the analysis timeline at 15 FPS.
    collision_timestamp = (
        collision_frame
        /
        TARGET_FPS
    )

    return {

        "collision_detected":
            "YES",

        "collision_frame":
            collision_frame,

        "collision_timestamp":
            format_timestamp(
                collision_timestamp
            ),

        "collision_distance":
            round(
                best_candidate[
                    "distance"
                ],
                2
            ),

        "vehicle_1":
            best_candidate[
                "vehicle_1"
            ],

        "vehicle_2":
            best_candidate[
                "vehicle_2"
            ],

        "close_frame_count":
            best_candidate[
                "close_count"
            ]
    }


# ================================================================
# PROCESS ONE VIDEO
# ================================================================

def process_video(
    video_path,
    model,
    video_number,
    total_videos
):

    print()
    print(
        "=" * 70
    )

    print(
        f"Processing video "
        f"{video_number}/{total_videos}"
    )

    print(
        "Video:",
        video_path.name
    )

    print(
        "=" * 70
    )

    info = get_video_info(
        video_path
    )

    original_fps = info[
        "original_fps"
    ]

    original_frame_count = info[
        "frame_count"
    ]

    original_width = info[
        "width"
    ]

    original_height = info[
        "height"
    ]

    duration = info[
        "duration"
    ]

    print(
        f"Original FPS       : "
        f"{original_fps:.2f}"
    )

    print(
        f"Original resolution: "
        f"{original_width}x"
        f"{original_height}"
    )

    print(
        f"Original frames    : "
        f"{original_frame_count}"
    )

    print(
        f"Duration           : "
        f"{duration:.2f} sec"
    )

    print(
        f"Analysis FPS       : "
        f"{TARGET_FPS:.2f}"
    )

    print(
        f"YOLO input         : "
        f"{YOLO_WIDTH}x"
        f"{YOLO_HEIGHT}"
    )

    # ------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    vehicle_history = {}

    trajectory_rows = []

    processed_frames = 0

    total_read_frames = 0

    last_progress = -1

    # ------------------------------------------------------------
    # Process frames
    # ------------------------------------------------------------

    while True:

        success, frame = (
            cap.read()
        )

        if not success:
            break

        total_read_frames += 1

        original_frame_index = (
            total_read_frames - 1
        )

        # --------------------------------------------------------
        # 15 FPS sampling
        # --------------------------------------------------------

        if not should_process_frame(
            original_frame_index,
            original_fps,
            TARGET_FPS
        ):

            continue

        processed_frames += 1

        # --------------------------------------------------------
        # ORIGINAL video timestamp
        # --------------------------------------------------------

        if original_fps > 0:

            original_timestamp = (
                original_frame_index
                /
                original_fps
            )

        else:

            original_timestamp = (
                processed_frames
                /
                TARGET_FPS
            )

        # --------------------------------------------------------
        # Resize to 640x640
        # --------------------------------------------------------

        resized_frame = cv2.resize(
            frame,
            (
                YOLO_WIDTH,
                YOLO_HEIGHT
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # --------------------------------------------------------
        # YOLO tracking
        # --------------------------------------------------------

        results = model.track(
            source=resized_frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE,
            classes=VEHICLE_CLASSES,
            verbose=False
        )

        if not results:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        if len(result.boxes) == 0:
            continue

        boxes = result.boxes

        xyxy = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

        if boxes.id is None:
            continue

        track_ids = (
            boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        # --------------------------------------------------------
        # Save trajectories
        # --------------------------------------------------------

        for box, track_id in zip(
            xyxy,
            track_ids
        ):

            x1, y1, x2, y2 = box

            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0

            vehicle_id = int(
                track_id
            )

            if vehicle_id not in vehicle_history:

                vehicle_history[
                    vehicle_id
                ] = []

            point = {

                "frame":
                    processed_frames - 1,

                "time":
                    original_timestamp,

                "x":
                    float(center_x),

                "y":
                    float(center_y)
            }

            vehicle_history[
                vehicle_id
            ].append(
                point
            )

            trajectory_rows.append({

                "video_name":
                    video_path.name,

                "vehicle_id":
                    vehicle_id,

                "analysis_frame":
                    processed_frames - 1,

                "original_frame":
                    original_frame_index,

                "timestamp_seconds":
                    round(
                        original_timestamp,
                        3
                    ),

                "timestamp":
                    format_timestamp(
                        original_timestamp
                    ),

                "center_x":
                    round(
                        float(center_x),
                        2
                    ),

                "center_y":
                    round(
                        float(center_y),
                        2
                    )
            })

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        if duration > 0:

            progress = int(
                (
                    original_timestamp
                    /
                    duration
                )
                * 100
            )

            progress = min(
                100,
                max(
                    0,
                    progress
                )
            )

            if progress != last_progress:

                last_progress = progress

                print(
                    f"\rProgress: "
                    f"{progress:3d}% "
                    f"| sampled frames: "
                    f"{processed_frames}",
                    end=""
                )

    cap.release()

    print()

    # ------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------

    collision = analyze_collision(
        vehicle_history
    )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    if (
        collision[
            "collision_detected"
        ]
        ==
        "YES"
    ):

        prediction = "ACCIDENT"

    else:

        prediction = "NON_ACCIDENT"

    # ------------------------------------------------------------
    # Save trajectory
    # ------------------------------------------------------------

    TRAJECTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trajectory_file = (
        TRAJECTORY_DIR
        /
        f"{video_path.stem}"
        f"_trajectory.csv"
    )

    trajectory_columns = [

        "video_name",
        "vehicle_id",
        "analysis_frame",
        "original_frame",
        "timestamp_seconds",
        "timestamp",
        "center_x",
        "center_y"
    ]

    if trajectory_rows:

        trajectory_df = (
            pd.DataFrame(
                trajectory_rows
            )
        )

    else:

        trajectory_df = pd.DataFrame(
            columns=trajectory_columns
        )

    trajectory_df.to_csv(
        trajectory_file,
        index=False
    )

    # ------------------------------------------------------------
    # Trajectory statistics
    # ------------------------------------------------------------

    vehicle_count = len(
        vehicle_history
    )

    trajectory_points = len(
        trajectory_rows
    )

    max_movement = 0.0

    total_movement = 0.0

    movement_count = 0

    for vehicle_id, points in (
        vehicle_history.items()
    ):

        if len(points) < 2:
            continue

        for k in range(
            1,
            len(points)
        ):

            p1 = points[
                k - 1
            ]

            p2 = points[
                k
            ]

            movement = center_distance(
                (
                    p1["x"],
                    p1["y"]
                ),
                (
                    p2["x"],
                    p2["y"]
                )
            )

            total_movement += movement

            movement_count += 1

            max_movement = max(
                max_movement,
                movement
            )

    if movement_count > 0:

        average_movement = (
            total_movement
            /
            movement_count
        )

    else:

        average_movement = 0.0

    # ------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------

    print(
        f"Prediction: {prediction}"
    )

    print(
        "Collision:",
        collision[
            "collision_detected"
        ]
    )

    if (
        collision[
            "collision_detected"
        ]
        ==
        "YES"
    ):

        print(
            "Collision frame:",
            collision[
                "collision_frame"
            ]
        )

        print(
            "Collision timestamp:",
            collision[
                "collision_timestamp"
            ]
        )

    print(
        "Vehicles tracked:",
        vehicle_count
    )

    # ------------------------------------------------------------
    # Result dictionary
    # ------------------------------------------------------------

    return {

        "Video Name":
            video_path.name,

        "Prediction":
            prediction,

        "Collision Detected":
            collision[
                "collision_detected"
            ],

        "Collision Frame":
            collision[
                "collision_frame"
            ],

        "Collision Timestamp":
            collision[
                "collision_timestamp"
            ],

        "Collision Distance (px)":
            collision[
                "collision_distance"
            ],

        "Vehicle 1":
            collision[
                "vehicle_1"
            ],

        "Vehicle 2":
            collision[
                "vehicle_2"
            ],

        "Close Frames":
            collision[
                "close_frame_count"
            ],

        "Original FPS":
            round(
                original_fps,
                3
            ),

        "Analysis FPS":
            TARGET_FPS,

        "Original Resolution":
            f"{original_width}x"
            f"{original_height}",

        "YOLO Resolution":
            f"{YOLO_WIDTH}x"
            f"{YOLO_HEIGHT}",

        "Original Frame Count":
            original_frame_count,

        "Processed Frames":
            processed_frames,

        "Duration (sec)":
            round(
                duration,
                3
            ),

        "Vehicles Tracked":
            vehicle_count,

        "Trajectory Points":
            trajectory_points,

        "Average Movement (px/frame)":
            round(
                average_movement,
                3
            ),

        "Maximum Movement (px/frame)":
            round(
                max_movement,
                3
            ),

        "Trajectory CSV":
            str(
                trajectory_file
            )
    }


# ================================================================
# CONFUSION MATRIX
# ================================================================

def calculate_metrics(
    results_df
):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for _, row in (
        results_df.iterrows()
    ):

        gt = str(
            row.get(
                "Ground Truth",
                ""
            )
        ).strip().upper()

        pred = str(
            row.get(
                "Prediction",
                ""
            )
        ).strip().upper()

        if (
            gt == "ACCIDENT"
            and
            pred == "ACCIDENT"
        ):

            tp += 1

        elif (
            gt == "NON_ACCIDENT"
            and
            pred == "NON_ACCIDENT"
        ):

            tn += 1

        elif (
            gt == "NON_ACCIDENT"
            and
            pred == "ACCIDENT"
        ):

            fp += 1

        elif (
            gt == "ACCIDENT"
            and
            pred == "NON_ACCIDENT"
        ):

            fn += 1

    total = (
        tp + tn + fp + fn
    )

    if total > 0:

        accuracy = (
            (tp + tn)
            /
            total
        )

    else:

        accuracy = 0.0

    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            /
            (tp + fp)
        )

    else:

        precision = 0.0

    if (
        tp + fn
    ) > 0:

        recall = (
            tp
            /
            (tp + fn)
        )

    else:

        recall = 0.0

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
        )

    else:

        f1 = 0.0

    return {

        "TP": tp,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "Total": total,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1
    }


# ================================================================
# CONFUSION LABEL FOR EACH VIDEO
# ================================================================

def get_confusion_label(
    ground_truth,
    prediction
):

    gt = str(
        ground_truth
    ).strip().upper()

    pred = str(
        prediction
    ).strip().upper()

    if (
        gt == "ACCIDENT"
        and
        pred == "ACCIDENT"
    ):

        return "TP"

    if (
        gt == "NON_ACCIDENT"
        and
        pred == "NON_ACCIDENT"
    ):

        return "TN"

    if (
        gt == "NON_ACCIDENT"
        and
        pred == "ACCIDENT"
    ):

        return "FP"

    if (
        gt == "ACCIDENT"
        and
        pred == "NON_ACCIDENT"
    ):

        return "FN"

    return "UNKNOWN"


# ================================================================
# SAVE EXCEL
# ================================================================

def save_excel(
    results_df,
    metrics,
    ground_truth_df
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------

    metrics_df = pd.DataFrame([

        {
            "Metric":
                "True Positive (TP)",

            "Value":
                metrics["TP"]
        },

        {
            "Metric":
                "True Negative (TN)",

            "Value":
                metrics["TN"]
        },

        {
            "Metric":
                "False Positive (FP)",

            "Value":
                metrics["FP"]
        },

        {
            "Metric":
                "False Negative (FN)",

            "Value":
                metrics["FN"]
        },

        {
            "Metric":
                "Total Videos Compared",

            "Value":
                metrics["Total"]
        },

        {
            "Metric":
                "Accuracy",

            "Value":
                round(
                    metrics["Accuracy"],
                    4
                )
        },

        {
            "Metric":
                "Precision",

            "Value":
                round(
                    metrics["Precision"],
                    4
                )
        },

        {
            "Metric":
                "Recall",

            "Value":
                round(
                    metrics["Recall"],
                    4
                )
        },

        {
            "Metric":
                "F1 Score",

            "Value":
                round(
                    metrics["F1 Score"],
                    4
                )
        }
    ])

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    confusion_df = pd.DataFrame({

        "":
            [
                "Actual ACCIDENT",
                "Actual NON_ACCIDENT"
            ],

        "Predicted ACCIDENT":
            [
                metrics["TP"],
                metrics["FP"]
            ],

        "Predicted NON_ACCIDENT":
            [
                metrics["FN"],
                metrics["TN"]
            ]
    })

    # ------------------------------------------------------------
    # Write workbook
    # ------------------------------------------------------------

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        results_df.to_excel(
            writer,
            sheet_name="Video Results",
            index=False
        )

        metrics_df.to_excel(
            writer,
            sheet_name="Performance Metrics",
            index=False
        )

        confusion_df.to_excel(
            writer,
            sheet_name="Confusion Matrix",
            index=False
        )

        ground_truth_df.to_excel(
            writer,
            sheet_name="Ground Truth",
            index=False
        )

    # ------------------------------------------------------------
    # Format workbook
    # ------------------------------------------------------------

    try:

        from openpyxl import load_workbook

        workbook = load_workbook(
            OUTPUT_EXCEL
        )

        for worksheet in (
            workbook.worksheets
        ):

            worksheet.freeze_panes = "A2"

            for column in (
                worksheet.columns
            ):

                max_length = 0

                column_letter = (
                    column[0].column_letter
                )

                for cell in column:

                    try:

                        length = len(
                            str(
                                cell.value
                            )
                        )

                        max_length = max(
                            max_length,
                            length
                        )

                    except Exception:
                        pass

                worksheet.column_dimensions[
                    column_letter
                ].width = min(
                    max_length + 2,
                    40
                )

        workbook.save(
            OUTPUT_EXCEL
        )

    except Exception as e:

        print()
        print(
            "WARNING: Excel formatting failed."
        )

        print(e)

    print()
    print(
        "Excel result created:"
    )

    print(
        OUTPUT_EXCEL
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "STAGE 1 - DATASET VIDEO AUTOMATION"
    )

    print(
        "=" * 70
    )

    print(
        "Project folder :",
        PROJECT_DIR
    )

    print(
        "Dataset folder :",
        DATASET_DIR
    )

    print(
        "Target FPS     :",
        TARGET_FPS
    )

    print(
        "YOLO input     :",
        f"{YOLO_WIDTH}x"
        f"{YOLO_HEIGHT}"
    )

    print(
        "Ground-truth comparison:"
        " ACCIDENT vs NON_ACCIDENT only"
    )

    print(
        "Timestamp comparison:"
        " NOT COMPARED"
    )

    print(
        "Trajectory comparison:"
        " NOT COMPARED"
    )

    # ------------------------------------------------------------
    # Check model
    # ------------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"YOLO model not found:\n"
            f"{MODEL_FILE}"
        )

    # ------------------------------------------------------------
    # Check dataset
    # ------------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset folder not found:\n"
            f"{DATASET_DIR}"
        )

    # ------------------------------------------------------------
    # Find videos
    # ------------------------------------------------------------

    videos = find_videos(
        DATASET_DIR
    )

    print()
    print(
        f"Videos found: "
        f"{len(videos)}"
    )

    if not videos:

        raise RuntimeError(
            "No videos found."
        )

    # ------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------

    print()
    print(
        "Ground truth file:",
        GROUND_TRUTH_FILE
    )

    (
        gt_mapping,
        gt_video_col,
        gt_label_col,
        gt_timestamp_col
    ) = load_ground_truth(
        GROUND_TRUTH_FILE
    )

    # ------------------------------------------------------------
    # Read original ground truth for Excel
    # ------------------------------------------------------------

    ground_truth_df = pd.read_excel(
        GROUND_TRUTH_FILE
    )

    # ------------------------------------------------------------
    # Show matching information
    # ------------------------------------------------------------

    print()
    print(
        "Ground-truth matching mode:"
    )

    print(
        "Extension-independent"
    )

    print(
        "Example:"
    )

    print(
        "video1 = video1.avi = video1.mp4"
    )

    # ------------------------------------------------------------
    # Load YOLO
    # ------------------------------------------------------------

    print()
    print(
        "Loading YOLO model..."
    )

    model = YOLO(
        str(MODEL_FILE)
    )

    print(
        "YOLO model loaded successfully."
    )

    # ------------------------------------------------------------
    # Process all videos
    # ------------------------------------------------------------

    all_results = []

    total_videos = len(
        videos
    )

    for video_number, video_path in enumerate(
        videos,
        start=1
    ):

        try:

            result = process_video(
                video_path,
                model,
                video_number,
                total_videos
            )

            # ----------------------------------------------------
            # AUTOMATIC GROUND-TRUTH MATCHING
            # ----------------------------------------------------

            video_key = normalize_video_name(
                video_path.name
            )

            if video_key in gt_mapping:

                gt_info = gt_mapping[
                    video_key
                ]

                ground_truth = normalize_label(
                    gt_info[
                        "label"
                    ]
                )

                ground_truth_timestamp = (
                    gt_info.get(
                        "timestamp",
                        ""
                    )
                )

                print(
                    "Ground truth matched:",
                    gt_info[
                        "original_name"
                    ],
                    "->",
                    ground_truth
                )

            else:

                ground_truth = ""

                ground_truth_timestamp = ""

                print()
                print(
                    "WARNING: Ground truth not found:"
                )

                print(
                    "Video:",
                    video_path.name
                )

                print(
                    "Matching key:",
                    video_key
                )

            # ----------------------------------------------------
            # Add ground truth
            # ----------------------------------------------------

            result[
                "Ground Truth"
            ] = ground_truth

            # Informational only.
            #
            # NOT used for TP/TN/FP/FN.

            result[
                "Ground Truth Timestamp"
            ] = ground_truth_timestamp

            # ----------------------------------------------------
            # TP/TN/FP/FN
            # ----------------------------------------------------

            result[
                "TP/TN/FP/FN"
            ] = get_confusion_label(
                ground_truth,
                result[
                    "Prediction"
                ]
            )

            all_results.append(
                result
            )

        except Exception as e:

            print()
            print(
                "ERROR processing:"
            )

            print(
                video_path.name
            )

            print(
                str(e)
            )

            traceback.print_exc()

            # ----------------------------------------------------
            # Try to preserve ground truth
            # ----------------------------------------------------

            video_key = normalize_video_name(
                video_path.name
            )

            gt_info = gt_mapping.get(
                video_key,
                {}
            )

            ground_truth = normalize_label(
                gt_info.get(
                    "label",
                    ""
                )
            )

            ground_truth_timestamp = (
                gt_info.get(
                    "timestamp",
                    ""
                )
            )

            all_results.append({

                "Video Name":
                    video_path.name,

                "Prediction":
                    "ERROR",

                "Collision Detected":
                    "ERROR",

                "Collision Frame":
                    "",

                "Collision Timestamp":
                    "",

                "Collision Distance (px)":
                    "",

                "Vehicle 1":
                    "",

                "Vehicle 2":
                    "",

                "Close Frames":
                    "",

                "Original FPS":
                    "",

                "Analysis FPS":
                    TARGET_FPS,

                "Original Resolution":
                    "",

                "YOLO Resolution":
                    f"{YOLO_WIDTH}x"
                    f"{YOLO_HEIGHT}",

                "Original Frame Count":
                    "",

                "Processed Frames":
                    "",

                "Duration (sec)":
                    "",

                "Vehicles Tracked":
                    "",

                "Trajectory Points":
                    "",

                "Average Movement (px/frame)":
                    "",

                "Maximum Movement (px/frame)":
                    "",

                "Trajectory CSV":
                    "",

                "Ground Truth":
                    ground_truth,

                "Ground Truth Timestamp":
                    ground_truth_timestamp,

                "TP/TN/FP/FN":
                    "ERROR"
            })

    # ------------------------------------------------------------
    # Create dataframe
    # ------------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    # ------------------------------------------------------------
    # Only valid prediction/ground truth rows
    # are used for metrics
    # ------------------------------------------------------------

    metric_rows = results_df[
        results_df[
            "Prediction"
        ].isin(
            [
                "ACCIDENT",
                "NON_ACCIDENT"
            ]
        )
        &
        results_df[
            "Ground Truth"
        ].isin(
            [
                "ACCIDENT",
                "NON_ACCIDENT"
            ]
        )
    ].copy()

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    metrics = calculate_metrics(
        metric_rows
    )

    # ------------------------------------------------------------
    # Put total metrics into main sheet
    # ------------------------------------------------------------

    results_df[
        "Total TP"
    ] = metrics[
        "TP"
    ]

    results_df[
        "Total TN"
    ] = metrics[
        "TN"
    ]

    results_df[
        "Total FP"
    ] = metrics[
        "FP"
    ]

    results_df[
        "Total FN"
    ] = metrics[
        "FN"
    ]

    results_df[
        "Accuracy"
    ] = round(
        metrics[
            "Accuracy"
        ],
        4
    )

    results_df[
        "Precision"
    ] = round(
        metrics[
            "Precision"
        ],
        4
    )

    results_df[
        "Recall"
    ] = round(
        metrics[
            "Recall"
        ],
        4
    )

    results_df[
        "F1 Score"
    ] = round(
        metrics[
            "F1 Score"
        ],
        4
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    save_excel(
        results_df,
        metrics,
        ground_truth_df
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "AUTOMATION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Videos found     : "
        f"{len(videos)}"
    )

    print(
        f"Videos processed : "
        f"{len(all_results)}"
    )

    print()

    print(
        f"TP: {metrics['TP']}"
    )

    print(
        f"TN: {metrics['TN']}"
    )

    print(
        f"FP: {metrics['FP']}"
    )

    print(
        f"FN: {metrics['FN']}"
    )

    print()

    print(
        f"Accuracy : "
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{metrics['Precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{metrics['Recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{metrics['F1 Score'] * 100:.2f}%"
    )

    print()

    print(
        "RESULT EXCEL:"
    )

    print(
        OUTPUT_EXCEL
    )

    print()

    print(
        "TRAJECTORY FOLDER:"
    )

    print(
        TRAJECTORY_DIR
    )

    print(
        "=" * 70
    )


# ================================================================
# START
# ================================================================

if __name__ == "__main__":

    main()