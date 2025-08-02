import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
import threading
import websocket
import json
from datetime import datetime, timedelta
import numpy as np
import platform
import sqlite3
import uuid
import os

class IMUGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU Real-Time Analysis System")
        self.fullscreen = False
        self.data_records = []
        self.pipeline = None
        self.streaming = False
        self.auto_mode = False
        self.collection_start_time = None
        self.threshold = 3.3
        self.session_id = None
        self.predictions_data = {}  # 예측 결과 저장용

        # 데이터베이스 초기화
        self.init_database()

        # 한글 폰트 설정
        self.setup_korean_font()
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8-whitegrid')
        
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        # 상단 컨트롤 패널
        self.setup_control_panel()
        
        # 상태 표시 패널
        self.setup_status_panel()
        
        # 그래프 설정
        self.setup_plots()
        
        # 하단 정보 패널
        self.setup_info_panel()

    def init_database(self):
        """데이터베이스 초기화"""
        self.db_path = "imu_analysis.db"
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS imu_raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    roll REAL NOT NULL,
                    pitch REAL NOT NULL,
                    yaw REAL NOT NULL,
                    x_del_ang REAL NOT NULL,
                    y_del_ang REAL NOT NULL,
                    z_del_ang REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS diagnosis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    measurement_date DATE NOT NULL,
                    measurement_time TIME NOT NULL,
                    data_collection_duration REAL NOT NULL,
                    predicted_roll_drift REAL,
                    predicted_pitch_drift REAL,
                    predicted_yaw_drift REAL,
                    max_drift_axis TEXT,
                    max_drift_value REAL,
                    max_drift_signed REAL,
                    is_faulty BOOLEAN NOT NULL,
                    fault_threshold REAL NOT NULL,
                    diagnosis_status TEXT NOT NULL,
                    model_version TEXT,
                    data_quality_score REAL,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS measurement_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    total_duration REAL,
                    sensor_count INTEGER,
                    total_data_points INTEGER,
                    session_type TEXT,
                    operator_name TEXT,
                    facility_location TEXT,
                    equipment_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("데이터베이스 초기화 완료")
            
        except Exception as e:
            print(f"데이터베이스 초기화 오류: {e}")
            messagebox.showerror("데이터베이스 오류", f"데이터베이스 초기화 실패:\n{e}")

    def setup_korean_font(self):
        """한글 폰트 설정"""
        system = platform.system()
        
        if system == 'Windows':
            font_candidates = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        elif system == 'Darwin':
            font_candidates = ['AppleGothic', 'Arial Unicode MS', 'DejaVu Sans']
        else:
            font_candidates = ['DejaVu Sans', 'Liberation Sans', 'Noto Sans CJK KR']
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        korean_font = None
        
        for font in font_candidates:
            if font in available_fonts:
                korean_font = font
                break
        
        if korean_font:
            plt.rcParams['font.family'] = korean_font
            self.font_family = korean_font
            print(f"한글 폰트 설정: {korean_font}")
        else:
            plt.rcParams['axes.unicode_minus'] = False
            self.font_family = 'DejaVu Sans'
            print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

    def setup_control_panel(self):
        # 메인 버튼 프레임
        btn_frame = tk.Frame(self.root, bg='lightgray', relief='raised', bd=2)
        btn_frame.pack(fill='x', pady=5, padx=10)
        
        # 큰 버튼들로 변경 - 한 줄에 배치
        button_config = {
            'font': (self.font_family, 12, 'bold'),
            'height': 2,
            'width': 15
        }
        
        tk.Button(btn_frame, text="자동 측정 시작", command=self.start_auto_collection, 
                 bg='green', fg='white', **button_config).pack(side='left', padx=5, pady=5)
        tk.Button(btn_frame, text="ML 모델 로드", command=self.load_model, 
                 bg='purple', fg='white', **button_config).pack(side='left', padx=5, pady=5)
        tk.Button(btn_frame, text="데이터 저장", command=self.save_data, 
                 bg='orange', fg='white', **button_config).pack(side='left', padx=5, pady=5)
        tk.Button(btn_frame, text="DB에 데이터 저장", command=self.save_to_database, 
                 bg='darkblue', fg='white', **button_config).pack(side='left', padx=5, pady=5)
        tk.Button(btn_frame, text="데이터 초기화", command=self.clear_data, 
                 bg='gray', fg='white', **button_config).pack(side='left', padx=5, pady=5)

    def setup_status_panel(self):
        # 상태 표시 프레임
        status_frame = tk.Frame(self.root, bg='lightblue', relief='sunken', bd=2)
        status_frame.pack(fill='x', pady=5, padx=10)
        
        self.status_label = tk.Label(status_frame, text="상태: 대기 중", 
                                   font=(self.font_family, 14, "bold"), bg='lightblue')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        self.countdown_label = tk.Label(status_frame, text="", 
                                      font=(self.font_family, 14, "bold"), bg='lightblue', fg='red')
        self.countdown_label.pack(side='right', padx=10, pady=5)

    def setup_plots(self):
        # 그래프 설정 - 더 큰 크기와 개선된 레이아웃
        self.fig, axs = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
        self.axes = axs.flatten()
        
        for i, ax in enumerate(self.axes):
            ax.set_title(f"센서 {i}", fontsize=16, fontweight='bold', fontfamily=self.font_family)
            ax.set_ylabel("각도 (도)", fontsize=12, fontfamily=self.font_family)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
        
        # X축 레이블은 하단 두 개만
        self.axes[6].set_xlabel("시간", fontsize=12, fontfamily=self.font_family)
        self.axes[7].set_xlabel("시간", fontsize=12, fontfamily=self.font_family)
        
        plt.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)

    def setup_info_panel(self):
        # 하단 정보 패널
        info_frame = tk.Frame(self.root, bg='lightyellow', relief='raised', bd=2)
        info_frame.pack(side='bottom', fill='x', padx=10, pady=5)
        
        thresh_label = tk.Label(info_frame, 
                              text=f"고장 판정 기준: {self.threshold}° 이상", 
                              font=(self.font_family, 16, "bold"), bg='lightyellow')
        thresh_label.pack(side='left', padx=10, pady=5)
        
        self.model_status = tk.Label(info_frame, text="ML 모델: 미로드", 
                                   font=(self.font_family, 14), bg='lightyellow', fg='red')
        self.model_status.pack(side='right', padx=10, pady=5)

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def update_status(self, message, color='black'):
        self.status_label.config(text=f"상태: {message}", fg=color)
        self.root.update()

    def start_auto_collection(self):
        """자동 측정 시작 - 5초 후 자동 종료 및 예측"""
        if self.pipeline is None:
            messagebox.showerror("오류", "먼저 ML 모델을 로드해주세요!")
            return
            
        self.auto_mode = True
        self.data_records = []  # 데이터 초기화
        self.predictions_data = {}  # 예측 결과 초기화
        self.collection_start_time = datetime.now()
        self.session_id = str(uuid.uuid4())  # 새 세션 ID 생성
        
        # 세션 정보 저장
        self.save_session_info()
        
        self.start_stream()
        self.update_status("자동 측정 중... (5초 후 자동 분석)", 'green')
        
        # 카운트다운 시작 (5초로 변경)
        self.start_countdown(5)

    def save_session_info(self):
        """측정 세션 정보를 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO measurement_sessions 
                (session_id, start_time, session_type, operator_name, facility_location, equipment_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.session_id,
                self.collection_start_time.isoformat(),
                "자동",
                "운영자",  # 필요시 사용자 입력으로 변경 가능
                "시설위치",  # 필요시 사용자 입력으로 변경 가능
                "IMU-001"  # 필요시 사용자 입력으로 변경 가능
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"세션 정보 저장 오류: {e}")

    def start_countdown(self, seconds_left):
        if seconds_left > 0 and self.auto_mode:
            self.countdown_label.config(text=f"남은 시간: {seconds_left}초")
            self.root.after(1000, lambda: self.start_countdown(seconds_left - 1))
        elif self.auto_mode:
            self.countdown_label.config(text="분석 중...")
            self.stop_stream()
            self.root.after(500, self.predict)  # 약간의 지연 후 예측 실행

    def clear_data(self):
        """데이터 초기화"""
        self.data_records = []
        self.predictions_data = {}
        for ax in self.axes:
            ax.cla()
            ax.set_title(f"센서 {self.axes.tolist().index(ax)}", fontsize=16, fontweight='bold', fontfamily=self.font_family)
            ax.set_ylabel("각도 (도)", fontsize=12, fontfamily=self.font_family)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
        self.canvas.draw()
        self.update_status("데이터 초기화 완료", 'blue')

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
            ts = datetime.now()
            if 'sensors' in msg:
                for sensor in msg['sensors']:
                    rec = sensor.copy()
                    rec['SN'] = rec.get('id')
                    rec['timestamp'] = ts
                    self.data_records.append(rec)
            else:
                msg['timestamp'] = ts
                self.data_records.append(msg)
        except Exception as e:
            print("메시지 파싱 오류:", e)

    def on_error(self, ws, error):
        print("WebSocket 오류:", error)
        self.update_status("연결 오류 발생", 'red')

    def on_close(self, ws, close_status, close_msg):
        print("WebSocket 연결 종료")
        if self.streaming:
            self.update_status("연결이 끊어졌습니다", 'red')

    def on_open(self, ws):
        print("WebSocket 연결 성공")
        self.update_status("데이터 수집 중", 'green')

    def start_stream(self):
        if self.streaming:
            return
        ws_url = "ws://192.168.62.16:81"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.streaming = True
        self.wst.start()
        self.root.after(100, self.update_plot)

    def stop_stream(self):
        if not self.streaming:
            return
        self.streaming = False
        self.ws.close()
        
        if self.auto_mode:
            self.countdown_label.config(text="")
            self.auto_mode = False
        else:
            self.update_status("데이터 수집 중지", 'orange')

    def update_plot(self):
        if not self.streaming:
            return
        
        if self.data_records:
            df = pd.DataFrame(self.data_records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            for ax in self.axes:
                ax.cla()
                
            for sn in range(8):
                ax = self.axes[sn]
                ax.set_title(f"센서 {sn}", fontsize=16, fontweight='bold', fontfamily=self.font_family)
                ax.set_facecolor('#f8f9fa')
                ax.grid(True, alpha=0.3)
                
                sub = df[df['SN'] == sn]
                if not sub.empty:
                    # 더 두꺼운 선과 명확한 색상
                    ax.plot(sub['timestamp'], sub['ROLL'], 'r-', linewidth=2, label='Roll', alpha=0.8)
                    ax.plot(sub['timestamp'], sub['PITCH'], 'g-', linewidth=2, label='Pitch', alpha=0.8)
                    ax.plot(sub['timestamp'], sub['YAW'], 'b-', linewidth=2, label='Yaw', alpha=0.8)
                    ax.legend(loc='upper right', fontsize=10, prop={'family': self.font_family})
                
                ax.set_ylabel("각도 (Deg)", fontsize=10, fontfamily=self.font_family)
                if sn >= 6:
                    ax.set_xlabel("시간", fontsize=10, fontfamily=self.font_family)
            
            plt.tight_layout(pad=2.0)
            self.canvas.draw()
        
        self.root.after(100, self.update_plot)

    def save_data(self):
        if not self.data_records:
            messagebox.showwarning("경고", "저장할 데이터가 없습니다")
            return
        
        df = pd.DataFrame(self.data_records)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel 파일","*.xlsx")]
        )
        if file_path:
            df.to_excel(file_path, index=False)
            messagebox.showinfo("성공", f"데이터가 저장되었습니다:\n{file_path}")

    def save_to_database(self):
        """수집된 데이터와 예측 결과를 데이터베이스에 저장"""
        if not self.data_records:
            messagebox.showwarning("경고", "저장할 데이터가 없습니다")
            return
        
        if not self.predictions_data:
            messagebox.showwarning("경고", "예측 결과가 없습니다. 먼저 자동 측정을 실행해주세요.")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. 원시 데이터 저장
            df = pd.DataFrame(self.data_records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            raw_data_count = 0
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT INTO imu_raw_data 
                    (session_id, sensor_id, timestamp, roll, pitch, yaw, x_del_ang, y_del_ang, z_del_ang)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.session_id,
                    int(row['SN']),
                    row['timestamp'].isoformat(),
                    float(row['ROLL']),
                    float(row['PITCH']),
                    float(row['YAW']),
                    float(row['X_DEL_ANG']),
                    float(row['Y_DEL_ANG']),
                    float(row['Z_DEL_ANG'])
                ))
                raw_data_count += 1
            
            # 2. 진단 결과 저장
            measurement_time = self.collection_start_time
            end_time = datetime.now()
            duration = (end_time - self.collection_start_time).total_seconds()
            
            diagnosis_count = 0
            for sensor_id, pred_info in self.predictions_data.items():
                cursor.execute('''
                    INSERT INTO diagnosis_results 
                    (session_id, sensor_id, measurement_date, measurement_time, data_collection_duration,
                     predicted_roll_drift, predicted_pitch_drift, predicted_yaw_drift,
                     max_drift_axis, max_drift_value, max_drift_signed,
                     is_faulty, fault_threshold, diagnosis_status, model_version, data_quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.session_id,
                    sensor_id,
                    measurement_time.date().isoformat(),
                    measurement_time.time().isoformat(),
                    duration,
                    pred_info.get('roll_drift'),
                    pred_info.get('pitch_drift'),
                    pred_info.get('yaw_drift'),
                    pred_info.get('max_drift_axis'),
                    pred_info.get('max_drift_value'),
                    pred_info.get('max_drift_signed'),
                    pred_info.get('is_faulty', False),
                    self.threshold,
                    pred_info.get('status', '정상'),
                    "v1.0",  # 모델 버전
                    1.0  # 데이터 품질 점수 (임시값)
                ))
                diagnosis_count += 1
            
            # 3. 세션 정보 업데이트
            active_sensors = len(set(df['SN']))
            cursor.execute('''
                UPDATE measurement_sessions 
                SET end_time = ?, total_duration = ?, sensor_count = ?, total_data_points = ?
                WHERE session_id = ?
            ''', (
                end_time.isoformat(),
                duration,
                active_sensors,
                len(df),
                self.session_id
            ))
            
            conn.commit()
            conn.close()
            
            messagebox.showinfo("성공", 
                f"데이터베이스에 저장 완료!\n"
                f"- 원시 데이터: {raw_data_count}개\n"
                f"- 진단 결과: {diagnosis_count}개\n"
                f"- 세션 ID: {self.session_id}")
            
        except Exception as e:
            messagebox.showerror("오류", f"데이터베이스 저장 실패:\n{e}")
            print(f"DB 저장 오류: {e}")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="ML 모델 파일 선택",
            filetypes=[("Pickle 파일","*.pkl")]
        )
        if not file_path:
            return
        
        try:
            self.pipeline = joblib.load(file_path)
            self.model_status.config(text="ML 모델: 로드 완료", fg='green')
            messagebox.showinfo("성공", "ML 모델이 성공적으로 로드되었습니다")
        except Exception as e:
            messagebox.showerror("오류", f"모델 로드 실패:\n{e}")

    def predict(self):
        if self.pipeline is None:
            messagebox.showerror("오류", "ML 모델이 로드되지 않았습니다")
            return
        
        if not self.data_records:
            messagebox.showwarning("경고", "예측할 데이터가 없습니다")
            return

        df = pd.DataFrame(self.data_records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        predictions = {}
        self.predictions_data = {}  # 데이터베이스 저장용 예측 결과

        for sn in range(8):
            sub = df[df['SN'] == sn].sort_values('timestamp')
            if len(sub) < 2:
                continue

            t0 = sub['timestamp'].iloc[0]
            start_time = t0 + timedelta(seconds=1)  # 1초 후부터
            end_time = start_time + timedelta(seconds=4)  # 4초간 (총 5초 수집)
            window = sub[(sub['timestamp'] >= start_time) & (sub['timestamp'] <= end_time)]
            
            if len(window) < 2:
                continue

            # 특성 계산
            p5 = (-window['X_DEL_ANG']).mean()
            q5 = (-window['Z_DEL_ANG']).mean()
            r5 = ( window['Y_DEL_ANG']).mean()

            dt = (window['timestamp'].iloc[-1] - window['timestamp'].iloc[0]).total_seconds()
            if dt == 0:
                continue
                
            Rd5 = (window['ROLL'].iloc[-1] - window['ROLL'].iloc[0]) / dt
            Pd5 = (window['PITCH'].iloc[-1] - window['PITCH'].iloc[0]) / dt
            Yd5 = (window['YAW'].iloc[-1] - window['YAW'].iloc[0]) / dt

            R = window['ROLL']
            P = window['PITCH']
            p = -window['X_DEL_ANG']
            q = -window['Z_DEL_ANG']
            r = window['Y_DEL_ANG']

            Rdot = p + (q * np.sin(np.deg2rad(R)) + r * np.cos(np.deg2rad(R))) * np.tan(np.deg2rad(P))
            Pdot = q * np.cos(np.deg2rad(R)) - r * np.sin(np.deg2rad(R))
            Ydot = (q * np.sin(np.deg2rad(R)) + r * np.cos(np.deg2rad(R))) / np.cos(np.deg2rad(P))

            Rdot5 = Rdot.mean()
            Pdot5 = Pdot.mean()
            Ydot5 = Ydot.mean()

            X_feat = pd.DataFrame([[p5, q5, r5, Rd5, Pd5, Yd5, Rdot5, Pdot5, Ydot5]],
                                  columns=['p5','q5','r5','Rd5','Pd5','Yd5','Rdot5','Pdot5','Ydot5'])
            try:
                raw_pred = self.pipeline.predict(X_feat)
                pred_vals = raw_pred[0] if hasattr(raw_pred[0], '__len__') else [raw_pred[0]]
            except Exception as e:
                messagebox.showerror("오류", f"센서 {sn} 예측 실패:\n{e}")
                return

            predictions[sn] = pred_vals
            
            # 데이터베이스 저장용 예측 결과 저장
            if len(pred_vals) == 3:
                r_pred, p_pred, y_pred = pred_vals
                drift_values = {'Roll': abs(r_pred), 'Pitch': abs(p_pred), 'Yaw': abs(y_pred)}
                max_drift_axis = max(drift_values, key=drift_values.get)
                max_drift_value = max(drift_values.values())
                
                if max_drift_axis == 'Roll':
                    max_drift_signed = r_pred
                elif max_drift_axis == 'Pitch':
                    max_drift_signed = p_pred
                else:
                    max_drift_signed = y_pred
                
                is_faulty = max_drift_value > self.threshold
                status = "고장" if is_faulty else "정상"
                
                self.predictions_data[sn] = {
                    'roll_drift': float(r_pred),
                    'pitch_drift': float(p_pred),
                    'yaw_drift': float(y_pred),
                    'max_drift_axis': max_drift_axis,
                    'max_drift_value': float(max_drift_value),
                    'max_drift_signed': float(max_drift_signed),
                    'is_faulty': is_faulty,
                    'status': status
                }
            else:
                # 단일 값 예측의 경우
                val = pred_vals[0]
                is_faulty = abs(val) > self.threshold
                status = "고장" if is_faulty else "정상"
                
                self.predictions_data[sn] = {
                    'roll_drift': None,
                    'pitch_drift': None,
                    'yaw_drift': None,
                    'max_drift_axis': 'Unknown',
                    'max_drift_value': float(abs(val)),
                    'max_drift_signed': float(val),
                    'is_faulty': is_faulty,
                    'status': status
                }

        # 예측 결과를 그래프에 표시
        self.display_predictions(predictions)
        
        if self.auto_mode:
            self.update_status("자동 분석 완료", 'blue')
        else:
            messagebox.showinfo("완료", "예측이 완료되었습니다")

    def display_predictions(self, predictions):
        """예측 결과를 그래프에 표시"""
        for sn, ax in enumerate(self.axes):
            pred = predictions.get(sn)
            
            # 기존 텍스트 제거
            for txt in ax.texts:
                txt.remove()
            
            if pred is None or len(pred) == 0:
                label = "데이터 부족"
                color = 'gray'
                bgcolor = 'lightgray'
            else:
                if len(pred) == 3:
                    # Roll, Pitch, Yaw 드리프트 예측값
                    r_pred, p_pred, y_pred = pred
                    
                    # 절댓값이 가장 큰 드리프트 찾기
                    drift_values = {'Roll': abs(r_pred), 'Pitch': abs(p_pred), 'Yaw': abs(y_pred)}
                    max_drift_axis = max(drift_values, key=drift_values.get)
                    max_drift_value = max(drift_values.values())
                    
                    # 원래 부호 유지
                    if max_drift_axis == 'Roll':
                        max_drift_signed = r_pred
                    elif max_drift_axis == 'Pitch':
                        max_drift_signed = p_pred
                    else:
                        max_drift_signed = y_pred
                    
                    fail = max_drift_value > self.threshold
                    status = "고장" if fail else "정상"
                    color = 'red' if fail else 'green'
                    bgcolor = 'mistyrose' if fail else 'lightgreen'
                    
                    label = f"100초 예측 드리프트\n최대: {max_drift_axis}\n{max_drift_signed:.2f}°\n상태: {status}"
                else:
                    # 단일 값 예측
                    val = pred[0]
                    fail = abs(val) > self.threshold
                    status = "고장" if fail else "정상"
                    color = 'red' if fail else 'green'
                    bgcolor = 'mistyrose' if fail else 'lightgreen'
                    label = f"100초 예측 드리프트\n{val:.2f}°\n상태: {status}"
            
            # 텍스트 박스 스타일로 결과 표시
            ax.text(0.5, 0.95, label, transform=ax.transAxes,
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bgcolor, alpha=0.8),
                    color=color, fontfamily=self.font_family)

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = IMUGUI(root)
    root.mainloop()