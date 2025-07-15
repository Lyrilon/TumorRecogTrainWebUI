import subprocess
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import eventlet

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
    cmd = ["python", "TumorTraining.py"]
    cmd.extend(["--num-workers", str(params.get("num_workers", 4))])
    cmd.extend(["--batch-size", str(params.get("batch_size", 32))])
    cmd.extend(["--epochs", str(params.get("epochs", 5))])

    for flag in [
        "balanced",
        "augmented",
        "augment-flip",
        "augment-offset",
        "augment-scale",
        "augment-rotate",
        "augment-noise",
    ]:
        if params.get(flag):
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


# --- 新增的推理功能路由 ---


@app.route("/start_inference", methods=["POST"])
def start_inference():
    """开始推理的路由"""
    global inference_process
    if inference_process and inference_process.poll() is None:
        inference_process.terminate()
        inference_process.wait()

    # 在实际应用中，您会从 request 中获取模型文件名
    # data = request.get_json()
    # model_file = data.get('model_file')
    # cmd = ['python', 'train.py', '--eval', model_file]

    # 为了模拟，我们使用一个固定的命令
    cmd = ["python", "train.py", "--eval", "dummy_model.pth"]
    print(f"Starting inference with command: {' '.join(cmd)}")
    inference_process = subprocess.Popen(cmd)
    return jsonify({"status": "Inference started"})


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
