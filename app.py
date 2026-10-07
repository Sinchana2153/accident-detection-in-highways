from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
from werkzeug.utils import secure_filename
from detector_adapter import detect_accident

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        flash("Please select a video.")
        return redirect(url_for("index"))

    video = request.files["video"]

    if not video.filename:
        flash("Please select a video.")
        return redirect(url_for("index"))

    if not allowed_file(video.filename):
        flash("Unsupported video format.")
        return redirect(url_for("index"))

    filename = secure_filename(video.filename)
    video_path = UPLOAD_DIR / filename
    video.save(video_path)

    try:
        result = detect_accident(str(video_path))
    except Exception as exc:
        return render_template(
            "result.html",
            filename=filename,
            result=None,
            error=f"Detection pipeline error: {exc}",
            video_url=url_for("uploaded_video", filename=filename),
        )

    return render_template(
        "result.html",
        filename=filename,
        result=result,
        error=None,
        video_url=url_for("uploaded_video", filename=filename),
    )

@app.route("/uploads/<path:filename>")
def uploaded_video(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/snapshots/<path:filename>")
def static_snapshot(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR / "snapshots", filename)

if __name__ == "__main__":
    app.run(debug=True)
