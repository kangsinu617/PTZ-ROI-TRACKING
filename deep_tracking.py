import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import threading
import time

# 설정
RTSP_URL = "rtsp://admin:password@ip주소:554/profile2/media.smp" # 비디오 스트림 링크
MODEL_PATH = "best_model_ptz.pth" # 학습 모델
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AI 튜닝값
NORM_FACTOR_ANGLE = 20.0
BASE_PAN_SENS = 15.0
BASE_TILT_SENS = 110.0

# Template matching
MATCH_THRESH = 0.60
LOCAL_MARGIN = 120
TEMPLATE_UPDATE_THRESH = 0.80

# LK Optical Flow 설정
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# 모델 정의
class SiameseNetworkPTZ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for name, param in self.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim * 3, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )

    def forward_one(self, x):
        return self.backbone(x)

    def forward(self, img1, img2):
        f1 = self.forward_one(img1)
        f2 = self.forward_one(img2)
        return self.regressor(torch.cat([f1, f2, torch.abs(f1 - f2)], 1))


# 3. 비디오 스레드
class VideoReader(threading.Thread):
    def __init__(self, rtsp_url):
        super().__init__()
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


# 4. 초기화
print(" 모델 로딩...")
model = SiameseNetworkPTZ().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
except:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(" 완료")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

video_thread = VideoReader(RTSP_URL)
video_thread.start()

# 상태 변수
prev_frame_tensor = None
old_gray = None

roi_points = None
roi_w, roi_h = 200, 200

template = None
is_tracking = False


def clamp_roi_rect(x, y, w, h, W, H):
    x = max(0, min(int(x), W - w))
    y = max(0, min(int(y), H - h))
    return x, y

def points_to_rect(pts):
    p = pts.reshape(-1, 2)
    x_min = np.min(p[:, 0])
    y_min = np.min(p[:, 1])
    x_max = np.max(p[:, 0])
    y_max = np.max(p[:, 1])
    return float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)

def rect_to_points(x, y, w, h):
    pts = np.array([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h]
    ], dtype=np.float32)
    return pts.reshape(-1, 1, 2)

def shift_points(pts, dx, dy):
    out = pts.copy()
    out[:, 0, 0] += dx
    out[:, 0, 1] += dy
    return out

def center_of_points(pts):
    p = pts.reshape(-1, 2)
    return float(np.mean(p[:, 0])), float(np.mean(p[:, 1]))


# ROI 설정
def mouse_callback(event, x, y, flags, param):
    global roi_points, prev_frame_tensor, old_gray, is_tracking, template

    if event == cv2.EVENT_LBUTTONDOWN:
        frame = video_thread.read()
        if frame is None:
            return

        H, W = frame.shape[:2]
        cx, cy = float(x), float(y)

        pts = np.array([
            [cx - roi_w/2, cy - roi_h/2],
            [cx + roi_w/2, cy - roi_h/2],
            [cx + roi_w/2, cy + roi_h/2],
            [cx - roi_w/2, cy + roi_h/2]
        ], dtype=np.float32)
        roi_points = pts.reshape(-1, 1, 2)

        # AI prev 텐서
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prev_frame_tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rx, ry, rw, rh = points_to_rect(roi_points)
        rx, ry = clamp_roi_rect(rx, ry, roi_w, roi_h, W, H)
        template = old_gray[ry:ry + roi_h, rx:rx + roi_w].copy()

        is_tracking = True
        print("ROI initialized (4 points + template saved)")


cv2.namedWindow("AI+LK+Template Polygon Tracker")
cv2.setMouseCallback("AI+LK+Template Polygon Tracker", mouse_callback)

print(" 영상 대기...")
while video_thread.read() is None:
    time.sleep(0.1)
print(" 준비 완료! ROI 중심을 클릭하세요. (ESC 종료)")


# 메인 루프
try:
    while True:
        frame = video_thread.read()
        if frame is None:
            continue

        display = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = frame.shape[:2]

        if is_tracking and roi_points is not None and old_gray is not None and template is not None:

            rgb_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            curr_tensor = transform(Image.fromarray(rgb_curr)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(prev_frame_tensor, curr_tensor)

            d_pan  = output[0, 0].item() * NORM_FACTOR_ANGLE
            d_tilt = output[0, 1].item() * NORM_FACTOR_ANGLE

            ai_dx = -(d_pan  * BASE_PAN_SENS)
            ai_dy = -(d_tilt * BASE_TILT_SENS)

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, roi_points, None, **lk_params)

            used_mode = "AI"
            color = (0, 255, 255)  # 노랑
            status = "AI only (fallback)"

            if p1 is not None and st is not None:
                st = st.reshape(-1)
                good_new = p1[st == 1]

                if good_new.shape[0] == 4:
                    roi_points = good_new.reshape(-1, 1, 2).astype(np.float32)
                    used_mode = "LK"
                    color = (0, 255, 0)  # 초록
                    status = "Tracking (LK)"
                else:
                    # LK가 깨졌으면 -> 템플릿 매칭으로 복구 시도
                    used_mode = "TEMPLATE/AI"
                    color = (0, 0, 255)  # 빨강
                    status = "Lost LK → Try Template"

                    pred_pts = shift_points(roi_points, ai_dx, ai_dy)
                    cx, cy = center_of_points(pred_pts)
                    pred_x = int(cx - roi_w // 2)
                    pred_y = int(cy - roi_h // 2)

                    best_loc = None
                    best_val = 0.0

                    sx1 = max(0, pred_x - LOCAL_MARGIN)
                    sy1 = max(0, pred_y - LOCAL_MARGIN)
                    sx2 = min(W, pred_x + roi_w + LOCAL_MARGIN)
                    sy2 = min(H, pred_y + roi_h + LOCAL_MARGIN)

                    if (sx2 - sx1) >= roi_w and (sy2 - sy1) >= roi_h:
                        local = frame_gray[sy1:sy2, sx1:sx2]
                        res = cv2.matchTemplate(local, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val >= MATCH_THRESH:
                            best_val = max_val
                            best_loc = (sx1 + max_loc[0], sy1 + max_loc[1])

                    if best_loc is None:
                        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val >= MATCH_THRESH:
                            best_val = max_val
                            best_loc = (max_loc[0], max_loc[1])

                    if best_loc is not None:
                        fx, fy = best_loc
                        fx, fy = clamp_roi_rect(fx, fy, roi_w, roi_h, W, H)
                        roi_points = rect_to_points(fx, fy, roi_w, roi_h)

                        status = f"LOCKED (Template {best_val:.2f})"
                        color = (255, 0, 255)  # 보라

                        if best_val >= TEMPLATE_UPDATE_THRESH:
                            template = frame_gray[fy:fy + roi_h, fx:fx + roi_w].copy()
                    else:
                        roi_points = shift_points(roi_points, ai_dx, ai_dy)
                        status = "Template fail → AI Recover"
                        color = (0, 0, 255)

            # 시각화
            draw_pts = roi_points.astype(np.int32)
            cv2.polylines(display, [draw_pts], True, color, 3)
            for pt in draw_pts:
                cv2.circle(display, (pt[0][0], pt[0][1]), 5, (0, 255, 255), -1)

            cv2.putText(display, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 다음 프레임 준비
            old_gray = frame_gray.copy()
            prev_frame_tensor = curr_tensor

        else:
            cv2.putText(display, "Click ROI Center", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("AI+LK+Template Polygon Tracker", display)
        if cv2.waitKey(1) == 27:
            break

finally:
    video_thread.stop()
    cv2.destroyAllWindows()
