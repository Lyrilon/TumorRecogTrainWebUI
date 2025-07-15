import subprocess
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import eventlet
import os

# 使用 eventlet 协程库
eventlet.monkey_patch()


app = Flask(__name__)
# 设置一个密钥，用于保护 session 和其他安全相关的事务
app.config["SECRET_KEY"] = "your-secret-key!"
# 初始化 SocketIO，并允许异步模式
socketio = SocketIO(app, async_mode="eventlet")

# 全局变量来跟踪进程状态
training_process = None
inference_process = None
training_paused = False


@app.route("/")
def index():
    """渲染主页面"""
    return render_template("index.html")


@app.route("/start_training", methods=["POST"])
def start_training():
    """开始训练的路由"""
    global training_process, training_paused

    if training_process and training_process.poll() is None:
        training_process.terminate()
        training_process.wait()

    params = request.get_json()
    print(f"the param is{params}")
    cmd = ["python", "segmentTraining.py"]
    cmd.extend(["--num-workers", str(params.get("num_workers", 4))])
    cmd.extend(["--batch-size", str(params.get("batch_size", 32))])
    cmd.extend(["--epochs", str(params.get("epochs", 5))])

    for flag in [
        "balanced",
        "augmented",
        "augment_flip",
        "augment_offset",
        "augment_scale",
        "augment_rotate",
        "augment_noise",
    ]:
        if params.get(flag):
            print(f"{flag}存在,加入")
            cmd.append(f"--{flag}")

    print(f"Starting training with command: {' '.join(cmd)}")
    training_paused = False
    training_process = subprocess.Popen(cmd)

    return jsonify({"status": "Training started"})


@app.route("/toggle_pause", methods=["POST"])
def toggle_pause():
    """切换训练暂停/继续状态的路由"""
    global training_paused
    training_paused = not training_paused
    state = "paused" if training_paused else "resumed"
    print(f"Training state toggled to: {state}")
    return jsonify({"status": state})


@app.route("/get_pause_state", methods=["GET"])
def get_pause_state():
    """供训练脚本查询是否应该暂停"""
    return jsonify({"paused": training_paused})


@app.route("/update_progress", methods=["POST"])
def update_progress():
    """接收来自 train.py 的训练进度"""
    data = request.get_json()
    socketio.emit("progress_update", data)
    return jsonify({"status": "success"})


@app.route("/training_done", methods=["POST"])
def training_done():
    """接收训练完成信号"""
    global training_process
    training_process = None
    socketio.emit("training_finished", {"status": "Training finished"})
    print("Received training finished signal.")
    return jsonify({"status": "done"})


# 定义一个用于保存上传模型的文件夹
UPLOAD_FOLDER = "uploaded_models"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 确保 Flask 应用知道上传文件夹
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

inference_process = None  # 全局变量来跟踪推理进程


@app.route("/start_inference", methods=["POST"])
def start_inference():
    """开始推理的路由，现在接收模型文件"""
    global inference_process
    if inference_process and inference_process.poll() is None:
        print("Existing inference process found, terminating...")
        inference_process.terminate()
        inference_process.wait()
        print("Existing inference process terminated.")

    # 检查请求中是否有文件
    if "model_file" not in request.files:
        return jsonify({"status": "No model file part in the request"}), 400

    model_file = request.files["model_file"]

    # 如果文件名为空（用户没有选择文件）
    if model_file.filename == "":
        return jsonify({"status": "No selected model file"}), 400

    if model_file:
        filename = model_file.filename

        # 构建文件保存路径
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # 保存文件
        model_file.save(filepath)
        print(f"Model file saved to: {filepath}")

        # 构建推理命令，现在传递保存的文件路径
        cmd = ["python", "segInference.py", "--eval", filepath]

        print(f"Starting inference with command: {' '.join(cmd)}")
        try:
            inference_process = subprocess.Popen(cmd)
            return jsonify({"status": "推理成功启动", "model_path": filepath})
        except Exception as e:
            print(f"Failed to start inference process: {e}")
            return (
                jsonify({"status": f"启动推理进程失败: {str(e)}"}),
                500,
            )

    return jsonify({"status": "Unknown error during file upload"}), 500


@app.route("/update_inference_image", methods=["POST"])
def update_inference_image():
    """接收来自 train.py 的推理结果图片URL"""
    data = request.get_json()
    # 将图片URL通过WebSocket发送给前端
    socketio.emit("inference_image_update", data)
    print(f"Sent inference image URLs to frontend.")
    return jsonify({"status": "success"})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
