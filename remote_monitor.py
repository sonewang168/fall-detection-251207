"""
🛡️ 樂齡防傾倒監測系統 v2.0 - 遠端監控版
=======================================
在父母家中電腦運行，你可以遠端查看即時影像並接收 LINE 通知

功能：
- 即時影像串流（瀏覽器查看）
- AI 跌倒偵測
- LINE 通知 + 截圖
- Gemini AI 分析
- 定時截圖回報

使用方式：
1. 安裝套件：pip install flask opencv-python mediapipe requests --break-system-packages
2. 修改下方 CONFIG 設定
3. 執行：python remote_monitor.py
4. 遠端瀏覽：http://59.127.52.150:8085
"""

from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import requests
import base64
from datetime import datetime

# ==================== 設定區（請修改這裡）====================
CONFIG = {
    # LINE Bot 設定
    "line_token": "在這裡貼上你的 Channel Access Token",
    "line_user_id": "U76a912d913fb2ca1bf85a16ea60e4ad4",
    
    # Gemini AI 設定
    "gemini_api_key": "在這裡貼上你的 Gemini API Key",
    
    # ImgBB 設定（截圖上傳）
    "imgbb_api_key": "在這裡貼上你的 ImgBB API Key",
    
    # 偵測參數
    "angle_threshold": 35,      # 傾斜角度閾值
    "frame_threshold": 15,      # 連續異常幀數
    "cooldown_seconds": 60,     # 通知冷卻時間（秒）
    
    # 定時回報（小時，0=關閉）
    "report_interval_hours": 1,
    
    # 伺服器設定
    "host": "0.0.0.0",         # 允許外部連線
    "port": 8085,              # 改成 8085
    
    # 攝影機
    "camera_index": 0,
}
# ============================================================

app = Flask(__name__)
CORS(app)  # 允許所有跨域請求

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 全域變數
camera = None
is_running = False
current_frame = None
current_status = {"status": "waiting", "angle": 0, "message": "等待啟動"}
abnormal_frame_count = 0
last_alert_time = 0
last_report_time = time.time()
initial_head_height = None
head_height_history = []
alert_count = 0
frame_lock = threading.Lock()

# ==================== 攝影機與偵測 ====================
def calculate_torso_angle(shoulder_mid, hip_mid):
    dx = abs(shoulder_mid[0] - hip_mid[0])
    dy = abs(shoulder_mid[1] - hip_mid[1])
    if dy < 0.001:
        return 90
    return np.degrees(np.arctan(dx / dy))

def calculate_head_height(nose_y, hip_y):
    global initial_head_height, head_height_history
    diff = hip_y - nose_y
    head_height_history.append(diff)
    if len(head_height_history) > 30:
        head_height_history.pop(0)
    if initial_head_height is None and len(head_height_history) >= 15:
        initial_head_height = sum(head_height_history[:15]) / 15
    return diff / initial_head_height if initial_head_height else 1

def process_frame(frame):
    global abnormal_frame_count, last_alert_time, current_status, alert_count
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if not results.pose_landmarks:
        current_status = {"status": "searching", "angle": 0, "message": "🔍 搜尋中..."}
        cv2.putText(frame, "Searching...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return frame
    
    landmarks = results.pose_landmarks.landmark
    h, w = frame.shape[:2]
    
    ls, rs = landmarks[11], landmarks[12]
    lh, rh = landmarks[23], landmarks[24]
    nose = landmarks[0]
    
    if ls.visibility > 0.5 and rs.visibility > 0.5 and lh.visibility > 0.5 and rh.visibility > 0.5:
        shoulder_mid = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        hip_mid = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
        
        angle = calculate_torso_angle(shoulder_mid, hip_mid)
        head_height = calculate_head_height(nose.y, hip_mid[1])
        
        # 畫骨架
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 212, 255), thickness=2))
        
        # 畫軀幹中線
        sm_px = (int(shoulder_mid[0] * w), int(shoulder_mid[1] * h))
        hm_px = (int(hip_mid[0] * w), int(hip_mid[1] * h))
        
        is_abnormal = angle > CONFIG["angle_threshold"] or head_height < 0.5
        severity = "danger" if angle > CONFIG["angle_threshold"] * 1.5 or head_height < 0.5 else "warning"
        
        if is_abnormal:
            abnormal_frame_count += 1
            color = (0, 0, 255)
            
            if abnormal_frame_count >= CONFIG["frame_threshold"]:
                now = time.time()
                if now - last_alert_time > CONFIG["cooldown_seconds"]:
                    alert_count += 1
                    threading.Thread(target=trigger_alert, args=(frame.copy(), angle, severity), daemon=True).start()
                    last_alert_time = now
                current_status = {"status": "danger", "angle": angle, "message": f"🚨 危險！{angle:.1f}°"}
            else:
                pct = int(abnormal_frame_count / CONFIG["frame_threshold"] * 100)
                current_status = {"status": "warning", "angle": angle, "message": f"⚠️ 偵測中 {pct}%"}
                color = (0, 165, 255)
        else:
            abnormal_frame_count = max(0, abnormal_frame_count - 2)
            color = (0, 255, 0)
            current_status = {"status": "normal", "angle": angle, "message": f"😊 正常 {angle:.1f}°"}
        
        cv2.line(frame, sm_px, hm_px, color, 6)
        cv2.putText(frame, f"Angle: {angle:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Head: {head_height:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame

def camera_thread():
    global current_frame, is_running, camera, last_report_time
    
    camera = cv2.VideoCapture(CONFIG["camera_index"])
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not camera.isOpened():
        print("❌ 無法開啟攝影機！")
        return
    
    print("✅ 攝影機已啟動")
    is_running = True
    
    while is_running:
        ret, frame = camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            processed = process_frame(frame)
            with frame_lock:
                current_frame = processed.copy()
            
            # 定時回報
            if CONFIG["report_interval_hours"] > 0:
                if time.time() - last_report_time > CONFIG["report_interval_hours"] * 3600:
                    threading.Thread(target=send_scheduled_report, args=(frame.copy(),), daemon=True).start()
                    last_report_time = time.time()
        
        time.sleep(0.03)
    
    camera.release()

def generate_frames():
    while True:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

# ==================== 通知功能 ====================
def upload_to_imgbb(image):
    if not CONFIG["imgbb_api_key"] or CONFIG["imgbb_api_key"].startswith("在這裡"):
        return None
    try:
        _, buffer = cv2.imencode('.jpg', image)
        b64 = base64.b64encode(buffer).decode('utf-8')
        r = requests.post(f"https://api.imgbb.com/1/upload?key={CONFIG['imgbb_api_key']}", data={"image": b64}, timeout=30)
        data = r.json()
        if data.get("success"):
            return data["data"]["url"]
    except Exception as e:
        print(f"ImgBB 上傳失敗: {e}")
    return None

def analyze_with_gemini(image):
    if not CONFIG["gemini_api_key"] or CONFIG["gemini_api_key"].startswith("在這裡"):
        return "（未設定 Gemini）"
    try:
        _, buffer = cv2.imencode('.jpg', image)
        b64 = base64.b64encode(buffer).decode('utf-8')
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={CONFIG['gemini_api_key']}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [
                {"text": "請用繁體中文，簡短分析（50字內）照片中人物姿態安全狀況：1.姿勢 2.跌倒風險 3.建議"},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
            ]}]},
            timeout=30
        )
        data = r.json()
        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Gemini 分析失敗: {e}")
    return "（AI 分析失敗）"

def send_line_message(messages):
    if not CONFIG["line_token"] or CONFIG["line_token"].startswith("在這裡"):
        print("❌ LINE Token 未設定")
        return False
    try:
        r = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {CONFIG['line_token']}"},
            json={"to": CONFIG["line_user_id"], "messages": messages},
            timeout=30
        )
        if r.status_code == 200:
            print("✅ LINE 發送成功")
            return True
        else:
            print(f"❌ LINE 發送失敗: {r.status_code} {r.text}")
    except Exception as e:
        print(f"❌ LINE 發送錯誤: {e}")
    return False

def trigger_alert(frame, angle, severity):
    print(f"🚨 觸發警報！角度: {angle:.1f}°")
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    
    messages = []
    
    # 上傳截圖
    img_url = upload_to_imgbb(frame)
    if img_url:
        messages.append({"type": "image", "originalContentUrl": img_url, "previewImageUrl": img_url})
    
    # 文字訊息
    severity_text = "🚨 嚴重" if severity == "danger" else "⚠️ 中度"
    messages.append({
        "type": "text",
        "text": f"🚨 跌倒警示！\n\n⏰ {now}\n📐 傾斜角度: {angle:.1f}°\n⚡ 程度: {severity_text}\n\n請立即確認長輩安全！\n\n🛡️ 樂齡防傾倒系統"
    })
    
    send_line_message(messages)

def send_scheduled_report(frame):
    print("📸 發送定時回報...")
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    
    messages = []
    
    # 上傳截圖
    img_url = upload_to_imgbb(frame)
    if img_url:
        messages.append({"type": "image", "originalContentUrl": img_url, "previewImageUrl": img_url})
    
    # Gemini 分析
    analysis = analyze_with_gemini(frame)
    
    messages.append({
        "type": "text",
        "text": f"📸 定時現況回報\n\n⏰ {now}\n\n🤖 AI 分析：\n{analysis}\n\n🛡️ 樂齡防傾倒系統"
    })
    
    send_line_message(messages)

# ==================== 網頁介面 ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ 樂齡守護 - 遠端監控</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #0a0a1a; color: white; min-height: 100vh; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; background: linear-gradient(135deg, #00d4ff, #7b2cbf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 28px; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #888; margin-bottom: 20px; }
        .video-container { position: relative; background: #12122a; border-radius: 20px; overflow: hidden; box-shadow: 0 0 40px rgba(0, 212, 255, 0.2); }
        .video-container img { width: 100%; display: block; }
        .status-bar { display: flex; justify-content: space-between; padding: 15px; background: rgba(0,0,0,0.5); }
        .status { padding: 8px 16px; border-radius: 20px; font-weight: bold; }
        .status.normal { background: #00ff88; color: #000; }
        .status.warning { background: #ffcc00; color: #000; }
        .status.danger { background: #ff3366; color: #fff; animation: pulse 0.5s infinite; }
        .status.searching { background: #00d4ff; color: #000; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px; }
        .info-card { background: #12122a; border-radius: 15px; padding: 20px; text-align: center; }
        .info-value { font-size: 32px; font-weight: bold; color: #00d4ff; }
        .info-label { font-size: 12px; color: #888; margin-top: 5px; }
        .btn { width: 100%; padding: 15px; border: none; border-radius: 12px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 15px; }
        .btn-primary { background: linear-gradient(135deg, #00d4ff, #0099cc); color: white; }
        .btn-danger { background: linear-gradient(135deg, #ff3366, #cc0033); color: white; }
        .config-info { background: #12122a; border-radius: 15px; padding: 20px; margin-top: 20px; font-size: 14px; }
        .config-info h3 { color: #00d4ff; margin-bottom: 10px; }
        .config-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #222; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ 樂齡守護</h1>
        <p class="subtitle">遠端監控系統 v2.0</p>
        
        <div class="video-container">
            <img src="/video_feed" alt="即時影像">
            <div class="status-bar">
                <span class="status" id="statusBadge">連線中...</span>
                <span style="color: #888;" id="timeDisplay">--:--:--</span>
            </div>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="info-value" id="angleValue">--°</div>
                <div class="info-label">傾斜角度</div>
            </div>
            <div class="info-card">
                <div class="info-value" id="alertCount">0</div>
                <div class="info-label">今日警示</div>
            </div>
            <div class="info-card">
                <div class="info-value" id="uptime">--</div>
                <div class="info-label">運行時間</div>
            </div>
        </div>
        
        <button class="btn btn-primary" onclick="sendTestReport()">📸 立即回報</button>
        <button class="btn btn-danger" onclick="sendTestAlert()">🧪 測試警報</button>
        
        <div class="config-info">
            <h3>⚙️ 系統設定</h3>
            <div class="config-item"><span>角度閾值</span><span>{{ angle_threshold }}°</span></div>
            <div class="config-item"><span>通知冷卻</span><span>{{ cooldown }}秒</span></div>
            <div class="config-item"><span>定時回報</span><span>每{{ report_interval }}小時</span></div>
            <div class="config-item"><span>LINE 通知</span><span id="lineStatus">檢查中...</span></div>
            <div class="config-item"><span>Gemini AI</span><span id="geminiStatus">檢查中...</span></div>
            <div class="config-item"><span>ImgBB 截圖</span><span id="imgbbStatus">檢查中...</span></div>
        </div>
    </div>
    
    <script>
        const startTime = Date.now();
        
        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const badge = document.getElementById('statusBadge');
                    badge.textContent = data.message;
                    badge.className = 'status ' + data.status;
                    document.getElementById('angleValue').textContent = data.angle.toFixed(1) + '°';
                    document.getElementById('alertCount').textContent = data.alert_count;
                    
                    document.getElementById('lineStatus').textContent = data.line_ok ? '✅ 已設定' : '❌ 未設定';
                    document.getElementById('geminiStatus').textContent = data.gemini_ok ? '✅ 已設定' : '❌ 未設定';
                    document.getElementById('imgbbStatus').textContent = data.imgbb_ok ? '✅ 已設定' : '❌ 未設定';
                });
        }
        
        function updateTime() {
            const now = new Date();
            document.getElementById('timeDisplay').textContent = now.toLocaleTimeString('zh-TW');
            
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const h = Math.floor(elapsed / 3600);
            const m = Math.floor((elapsed % 3600) / 60);
            document.getElementById('uptime').textContent = h + '時' + m + '分';
        }
        
        function sendTestReport() {
            fetch('/api/report').then(r => r.json()).then(d => alert(d.message));
        }
        
        function sendTestAlert() {
            fetch('/api/test_alert').then(r => r.json()).then(d => alert(d.message));
        }
        
        setInterval(updateStatus, 1000);
        setInterval(updateTime, 1000);
        updateStatus();
        updateTime();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,
        angle_threshold=CONFIG["angle_threshold"],
        cooldown=CONFIG["cooldown_seconds"],
        report_interval=CONFIG["report_interval_hours"] or "關閉"
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        **current_status,
        "alert_count": alert_count,
        "line_ok": not CONFIG["line_token"].startswith("在這裡"),
        "gemini_ok": not CONFIG["gemini_api_key"].startswith("在這裡"),
        "imgbb_ok": not CONFIG["imgbb_api_key"].startswith("在這裡")
    })

@app.route('/api/report')
def api_report():
    with frame_lock:
        if current_frame is not None:
            threading.Thread(target=send_scheduled_report, args=(current_frame.copy(),), daemon=True).start()
            return jsonify({"status": "ok", "message": "📸 回報已發送！"})
    return jsonify({"status": "error", "message": "❌ 無法擷取畫面"})

@app.route('/api/test_alert')
def api_test_alert():
    with frame_lock:
        if current_frame is not None:
            threading.Thread(target=trigger_alert, args=(current_frame.copy(), 99.9, "danger"), daemon=True).start()
            return jsonify({"status": "ok", "message": "🚨 測試警報已發送！"})
    return jsonify({"status": "error", "message": "❌ 無法擷取畫面"})

# ==================== 啟動 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("🛡️ 樂齡防傾倒監測系統 v2.0 - 遠端監控版")
    print("=" * 50)
    print(f"📡 伺服器位址: http://0.0.0.0:{CONFIG['port']}")
    print(f"🌐 遠端存取: http://59.127.52.150:{CONFIG['port']}")
    print("=" * 50)
    
    # 啟動攝影機執行緒
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()
    
    # 啟動 Flask
    app.run(host=CONFIG["host"], port=CONFIG["port"], threaded=True, debug=False)
