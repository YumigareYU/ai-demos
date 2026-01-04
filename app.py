import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

# --- 頁面設定 ---
st.set_page_config(page_title="AI 影像辨識實驗室", page_icon="🤖")

st.title("🐱 貓咪滾球追蹤 (AI 辨識)")
st.write("上傳影片後，系統會自動進行追蹤處理。")

# --- 側邊欄或主畫面選項 ---
st.write("---")
use_demo = st.checkbox("👉 沒有影片嗎？點我直接使用「範例影片」試玩", value=False)

input_path = None
temp_file_to_clean = None  # 用來記錄是否需要刪除暫存檔

# --- 邏輯判斷：決定影片來源 ---
if use_demo:
    # 來源 A: 使用範例影片
    if os.path.exists("demo.mp4"):
        input_path = "demo.mp4"
        st.info("正在使用範例影片進行演示...")
    else:
        st.error("❌ 找不到 demo.mp4 檔案，請確認是否已上傳到 GitHub。")
else:
    # 來源 B: 使用者上傳
    uploaded_file = st.file_uploader(
        "或是選擇您自己的影片...", type=['mp4', 'mov', 'avi', 'webm'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()  # 關閉檔案以免 Windows 鎖定

        input_path = tfile.name
        temp_file_to_clean = tfile.name  # 標記這個檔案最後要刪除

# --- 3. 開始處理 (如果有取得路徑) ---
if input_path is not None:
    try:
        cap = cv2.VideoCapture(input_path)

        # 取得影片資訊
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 設定輸出檔案
        output_filename = "output_result.webm"
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        # --- 介面元件 ---
        st.write("🔄 AI 正在逐格分析影片中，請稍候...")
        my_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- 影像處理邏輯 ---
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            lower_pink = np.array([130, 50, 50])
            upper_pink = np.array([175, 255, 255])
            mask = cv2.inRange(hsv, lower_pink, upper_pink)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=2)
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
            # -------------------

            out.write(frame)

            frame_count += 1
            if total_frames > 0:
                my_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(
                f"Processing frame: {frame_count} / {total_frames}")

        cap.release()
        out.release()

        # --- 處理完成 ---
        my_bar.empty()
        status_text.empty()
        st.success("✅ 處理完成！")

        # 播放影片
        st.video(output_filename)

    finally:
        # 清理邏輯：只刪除上傳產生的暫存檔，不要刪除範例影片 demo.mp4
        time.sleep(1)
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            os.remove(temp_file_to_clean)
