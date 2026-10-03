from flask import Flask, render_template, abort, jsonify, request, redirect, url_for, flash
from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime

app = Flask(__name__)
app.secret_key = "deep-learning-practice-local"

BASE_DIR = Path(__file__).resolve().parent
EXERCISES_DIR = BASE_DIR / "exercises"
DATA_FILE = BASE_DIR / "data" / "exercises.json"
EXERCISES_DIR.mkdir(exist_ok=True)
DATA_FILE.parent.mkdir(exist_ok=True)

DEFAULT_TOPICS = {
    "1": {
        "title": "1-MAVZU",
        "subtitle": "Sun’iy neyron, perceptron va logistika neyronlari",
        "description": "Sun’iy neyronlar, perceptron, logistika neyroni va ularning matematik asoslari bo‘yicha amaliy mashg‘ulotlar.",
        "icon": "🧠"
    },
    "2": {
        "title": "2-MAVZU",
        "subtitle": "Backpropagation, Softmax va optimizatsiya algoritmlari",
        "description": "Backpropagation, Softmax, gradientlar, loss funksiyalari va optimizatorlar bo‘yicha chuqur amaliy topshiriqlar.",
        "icon": "⚡"
    }
}

def load_data():
    if not DATA_FILE.exists():
        return {"topics": DEFAULT_TOPICS.copy(), "exercises": {}}
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {"topics": DEFAULT_TOPICS.copy(), "exercises": {}}

    # Backward compatibility with the old flat exercises.json
    if "topics" not in data or "exercises" not in data:
        old = data if isinstance(data, dict) else {}
        data = {"topics": DEFAULT_TOPICS.copy(), "exercises": old}
        for item in data["exercises"].values():
            item.setdefault("topic_ids", [1, 2])
    data.setdefault("topics", DEFAULT_TOPICS.copy())
    data.setdefault("exercises", {})
    return data

def save_data(data):
    tmp = DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DATA_FILE)

def get_exercises():
    items = []
    for path in EXERCISES_DIR.glob("*.py"):
        m = re.match(r"^(\d+)\.py$", path.name)
        if m:
            items.append((int(m.group(1)), path))
    return sorted(items)

def topic_exercises(topic_id):
    data = load_data()
    available = dict(get_exercises())
    result = []
    for key, info in data["exercises"].items():
        try:
            n = int(key)
        except ValueError:
            continue
        if topic_id in [int(x) for x in info.get("topic_ids", [])] and n in available:
            result.append((n, available[n]))
    return sorted(result)

def next_exercise_number():
    nums = [n for n, _ in get_exercises()]
    return max(nums, default=0) + 1

def next_topic_id():
    data = load_data()
    ids = [int(k) for k in data["topics"].keys() if str(k).isdigit()]
    return max(ids, default=0) + 1

def get_context(number, requested_topic=None):
    data = load_data()
    valid_topics = {int(k): v for k, v in data["topics"].items()}
    topic_id = requested_topic if requested_topic in valid_topics else None

    if topic_id is None:
        info = data["exercises"].get(str(number), {})
        ids = info.get("topic_ids", [])
        topic_id = int(ids[0]) if ids else 1

    exs = topic_exercises(topic_id)
    numbers = [n for n, _ in exs]
    if number not in numbers:
        abort(404)
    idx = numbers.index(number)
    prev_n = numbers[idx-1] if idx > 0 else None
    next_n = numbers[idx+1] if idx + 1 < len(numbers) else None
    return data, valid_topics[topic_id], topic_id, data["exercises"].get(str(number), {}), prev_n, next_n

@app.context_processor
def inject_global():
    data = load_data()
    return {
        "topics": {int(k): v for k, v in data["topics"].items()},
        "current_year": datetime.now().year
    }

@app.route("/")
def index():
    data = load_data()
    topics = {int(k): v for k, v in data["topics"].items()}
    return render_template(
        "index.html",
        topic_exercises={tid: topic_exercises(tid) for tid in topics},
        metadata=data["exercises"]
    )

@app.route("/topic/<int:topic_id>")
def topic(topic_id):
    data = load_data()
    topic_data = data["topics"].get(str(topic_id))
    if not topic_data:
        abort(404)
    exs = topic_exercises(topic_id)
    return render_template("topic.html", topic_id=topic_id, topic=topic_data,
                           exercises=exs, metadata=data["exercises"])

@app.route("/exercise/<int:number>")
def exercise(number):
    path = EXERCISES_DIR / f"{number}.py"
    if not path.exists():
        abort(404)
    requested_topic = request.args.get("topic", type=int)
    data, topic_data, topic_id, info, prev_n, next_n = get_context(number, requested_topic)
    code = path.read_text(encoding="utf-8")
    return render_template(
        "exercise.html", number=number, code=code, info=info,
        exercises=topic_exercises(topic_id), previous_number=prev_n,
        next_number=next_n, topic_id=topic_id, topic=topic_data
    )

import zipfile


EXAMPLE_ZIPS = {
    "m01": BASE_DIR / "examples" / "m01_camera.zip",
    "m02": BASE_DIR / "examples" / "m02_digital.zip",
}

@app.post("/run-example/<example_key>")
def run_example(example_key):
    """m01_camera yoki m02_digital yechimini ishga tushirib,
    stdout va hosil bo'lgan barcha rasmlarni webga qaytaradi."""
    if example_key not in EXAMPLE_ZIPS:
        return jsonify({"ok": False, "output": "Misol topilmadi."}), 404

    archive = EXAMPLE_ZIPS[example_key]
    if not archive.exists():
        return jsonify({"ok": False, "output": "Misol ZIP fayli topilmadi."}), 404

    root = BASE_DIR / "static" / "generated_examples" / example_key
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"dl_{example_key}_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp_path)

        candidates = list(tmp_path.glob("*/solutions.py"))
        if not candidates:
            return jsonify({"ok": False, "output": "solutions.py topilmadi."}), 500
        project = candidates[0]

        try:
            result = subprocess.run(
                [sys.executable, str(project)],
                cwd=str(project.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120
            )
        except subprocess.TimeoutExpired:
            return jsonify({
                "ok": False,
                "output": "Misol 120 soniyadan ko‘p ishladi va to‘xtatildi.",
                "images": []
            }), 408
        except Exception as e:
            return jsonify({"ok": False, "output": f"Xatolik: {e}", "images": []}), 500

        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr

        images = []
        out_dir = project.parent / "out"
        if out_dir.exists():
            for image_path in sorted(out_dir.rglob("*")):
                if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    rel = image_path.relative_to(out_dir)
                    dest = root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, dest)
                    images.append(url_for(
                        "static",
                        filename=f"generated_examples/{example_key}/{rel.as_posix()}"
                    ))

        # m01 uchun yuklangan kirish rasmlaridan ham ko‘rsatamiz.
        if example_key == "m01":
            data_dir = project.parent / "data"
            for image_path in sorted(data_dir.glob("*.jpg")):
                dest = root / "input" / image_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, dest)
                images.append(url_for(
                    "static",
                    filename=f"generated_examples/{example_key}/input/{image_path.name}"
                ))

        return jsonify({
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "output": output if output else "(Natija chiqarilmadi.)",
            "images": images
        })

@app.post("/run/<int:number>")
def run_exercise(number):
    path = EXERCISES_DIR / f"{number}.py"
    if not path.exists():
        return jsonify({"ok": False, "output": "Fayl topilmadi."}), 404
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(EXERCISES_DIR),
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60
        )
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return jsonify({
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "output": output if output else "(Natija chiqarilmadi.)"
        })
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "output": "Xatolik: dastur 60 soniyadan ko‘p ishladi va to‘xtatildi."}), 408
    except Exception as e:
        return jsonify({"ok": False, "output": f"Xatolik: {e}"}), 500

# ---------------- ADMIN / CONTENT MANAGEMENT ----------------

@app.route("/admin")
def admin():
    data = load_data()
    topics = {int(k): v for k, v in data["topics"].items()}
    exercises = []
    for key, info in data["exercises"].items():
        try:
            number = int(key)
        except ValueError:
            continue
        exercises.append((number, info))
    exercises.sort(key=lambda item: item[0])
    return render_template(
        "admin.html",
        topics=topics,
        exercises=exercises,
        total_exercises=len(data["exercises"]),
        next_number=next_exercise_number(),
        next_topic=next_topic_id()
    )

@app.post("/admin/topic/add")
def add_topic():
    title = request.form.get("title", "").strip()
    subtitle = request.form.get("subtitle", "").strip()
    description = request.form.get("description", "").strip()
    icon = request.form.get("icon", "📚").strip() or "📚"

    if not title or not subtitle:
        flash("Mavzu nomi va sarlavhasi majburiy.", "error")
        return redirect(url_for("admin"))

    data = load_data()
    tid = next_topic_id()
    data["topics"][str(tid)] = {
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "icon": icon
    }
    save_data(data)
    flash(f"{title} muvaffaqiyatli qo‘shildi.", "success")
    return redirect(url_for("admin"))

@app.post("/admin/topic/delete/<int:topic_id>")
def delete_topic(topic_id):
    data = load_data()
    key = str(topic_id)
    if key not in data["topics"]:
        abort(404)

    # Faqat mavzuning o‘zini o‘chiramiz. Masalalar va ularning
    # .py fayllari saqlanadi, boshqa mavzularga tegilmaydi.
    del data["topics"][key]

    # O‘chirilgan mavzu ID sini masalalardan ham olib tashlaymiz.
    # Agar masala boshqa mavzuga ham tegishli bo‘lsa, u saqlanadi.
    for info in data["exercises"].values():
        topic_ids = info.get("topic_ids", [])
        info["topic_ids"] = [int(t) for t in topic_ids if int(t) != topic_id]

    save_data(data)
    flash(f"{topic_id}-mavzu o‘chirildi. Masalalar saqlandi.", "success")
    return redirect(url_for("admin"))

@app.post("/admin/exercise/add")
def add_exercise():
    data = load_data()

    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    statement = request.form.get("statement", "").strip()
    code = request.form.get("code", "")
    topic_id = request.form.get("topic_id", type=int)
    formulas_raw = request.form.get("formulas", "")
    algorithm_raw = request.form.get("algorithm", "")
    details_formula = request.form.getlist("detail_formula")
    details_explanation = request.form.getlist("detail_explanation")
    details_code = request.form.getlist("detail_code")

    if topic_id not in [int(k) for k in data["topics"]]:
        flash("Mavzu tanlanmadi.", "error")
        return redirect(url_for("admin"))
    if not title or not code.strip():
        flash("Masala nomi va Python kodi majburiy.", "error")
        return redirect(url_for("admin"))

    number = next_exercise_number()
    formulas = [x.strip() for x in formulas_raw.splitlines() if x.strip()]
    algorithm = [x.strip() for x in algorithm_raw.splitlines() if x.strip()]

    formula_details = []
    for f, e, c in zip(details_formula, details_explanation, details_code):
        if f.strip():
            formula_details.append({
                "formula": f.strip(),
                "explanation": e.strip(),
                "python_code": c.strip()
            })

    file_path = EXERCISES_DIR / f"{number}.py"
    file_path.write_text(code.rstrip() + "\n", encoding="utf-8")

    data["exercises"][str(number)] = {
        "title": title,
        "description": description or "Python kodi va amaliy yechim.",
        "statement": statement,
        "formulas": formulas,
        "algorithm": algorithm,
        "formula_details": formula_details,
        "topic_ids": [topic_id]
    }
    save_data(data)
    flash(f"{number}-masala muvaffaqiyatli qo‘shildi.", "success")
    return redirect(url_for("exercise", number=number, topic=topic_id))

@app.post("/admin/exercise/delete/<int:number>")
def delete_exercise(number):
    data = load_data()
    key = str(number)
    if key not in data["exercises"]:
        abort(404)
    path = EXERCISES_DIR / f"{number}.py"
    if path.exists():
        path.unlink()
    del data["exercises"][key]
    save_data(data)
    flash(f"{number}-masala o‘chirildi.", "success")
    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True)
