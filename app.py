import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# --- 頁面設定 ---
st.set_page_config(page_title="AI 影像辨識實驗室", page_icon="🤖")

st.title("🐱 貓咪滾球追蹤 (AI 辨識)")
st.write("這是使用 **OpenCV** 與 **Streamlit** 建構的即時影像分析。請上傳貓咪玩粉紅球的影片。")

# --- 1. 上傳影片 ---
uploaded_file = st.file_uploader("請選擇影片檔案...", type=['mp4', 'mov', 'avi', 'webm'])

# --- 2. 開始處理 ---
if uploaded_file is not None:
    # 建立一個暫存檔來儲存上傳的影片 (因為 OpenCV 需要讀取實體檔案)
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    # 開啟影片
    cap = cv2.VideoCapture(tfile.name)
    
    # 建立一個空位，用來不斷更新畫面
    st_frame = st.empty()
    
    # 建立一個停止按鈕
    stop_button = st.button("停止播放")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- 您的 OpenCV 辨識邏輯 (原封不動搬過來) ---
        # 1. 模糊化
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        
        # 2. 轉換顏色空間 BGR -> HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 3. 定義粉紅色範圍
        lower_pink = np.array([130, 50, 50])
        upper_pink = np.array([175, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # 4. 消除雜訊
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.erode(mask, None, iterations=1)

        # 5. 尋找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100: continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

            if circularity > 0.6:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # 畫圓圈
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                # 寫文字
                cv2.putText(frame, f"Ball: {circularity:.2f}", (int(x), int(y)-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # ---------------------------------------------
        
        # --- 關鍵：將 BGR 轉回 RGB 才能在網頁正常顯示 ---
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 更新畫面
        st_frame.image(frame, channels="RGB", use_container_width=True)

    cap.release()
    # 刪除暫存檔
    os.remove(tfile.name)