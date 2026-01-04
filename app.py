import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# --- 頁面設定 ---
st.set_page_config(page_title="AI 影像辨識實驗室", page_icon="🤖")

st.title("🐱 貓咪滾球追蹤 (AI 辨識)")
st.write("上傳影片後，系統會自動進行追蹤處理，並產生流暢的結果影片。")

# --- 1. 上傳影片 ---
uploaded_file = st.file_uploader(
    "請選擇影片檔案...", type=['mp4', 'mov', 'avi', 'webm'])

# --- 2. 開始處理 ---
if uploaded_file is not None:
    # 建立暫存檔讀取上傳的影片
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    # 取得影片資訊 (為了製作進度條和設定輸出格式)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 設定輸出檔案 (使用 WebM + VP90 編碼，確保瀏覽器能播)
    output_path = tfile.name + "_output.webm"
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- 介面元件 ---
    st.write("🔄 AI 正在逐格分析影片中，請稍候...")
    my_bar = st.progress(0)  # 建立進度條
    status_text = st.empty()  # 顯示目前幀數

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- 您的 OpenCV 辨識邏輯 (維持不變) ---
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_pink = np.array([130, 50, 50])
        upper_pink = np.array([175, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.erode(mask, None, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

            if circularity > 0.6:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                cv2.circle(frame, (int(x), int(y)),
                           int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f"Ball: {circularity:.2f}", (int(x), int(y)-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # ---------------------------------------------

        # 寫入處理後的影片檔
        out.write(frame)

        # 更新進度條
        frame_count += 1
        if total_frames > 0:
            my_bar.progress(min(frame_count / total_frames, 1.0))
        status_text.text(f"Processing frame: {frame_count} / {total_frames}")

    cap.release()
    out.release()

    # --- 處理完成，顯示結果 ---
    my_bar.empty()     # 隱藏進度條
    status_text.empty()  # 隱藏文字
    st.success("✅ 處理完成！")

    # 播放影片
    st.video(output_path)

    # 清理暫存檔
    os.remove(tfile.name)
    # os.remove(output_path) # 這裡先不刪除，以免影片還沒看完就被刪掉
