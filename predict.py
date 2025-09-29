# ================================================================= #
# 단계 1: 기본 환경 준비 및 라이브러리 임포트
# ================================================================= #
# 필요한 라이브러리가 없다면 설치합니다. 예: !pip install wget -q
import sys
import os
import wget
import warnings
import torch
import torchio
import torch.nn as nn

# Colab 환경인지 확인하고 AST 소스 코드를 다운로드합니다.
# 모델 구조(ASTModel)를 불러오기 위해 이 과정은 필수적입니다.
if 'google.colab' in sys.modules:
    print('Colab 환경에서 실행 중입니다.')
    if not os.path.exists('/content/ast'):
        print('AST 소스 코드를 다운로드합니다...')
        !git clone https://github.com/YuanGongND/ast.git > /dev/null 2>&1
    sys.path.append('/content/ast')

# ASTModel을 사용하기 위해 src.models에서 import 합니다.
from src.models import ASTModel

warnings.filterwarnings('ignore')
print("\n기본 환경 준비가 완료되었습니다.")


# ================================================================= #
# 단계 2: 학습 때 사용했던 '헬퍼 함수'와 '모델 구조' 정의
# 참고: 이 코드들은 학습 코드와 100% 동일해야 합니다.
# ================================================================= #

# --- 오디오 전처리 함수 ---
def make_features(wav_name, mel_bins, target_length=1024):
    """학습 때와 동일한 방법으로 오디오 파일을 mel-spectrogram으로 변환합니다."""
    try:
        waveform, sr = torchaudio.load(wav_name)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10
        )
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0: fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0: fbank = fbank[0:target_length, :]
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2) # Normalization
        return fbank
    except Exception as e:
        print(f" - '{os.path.basename(wav_name)}' 파일 처리 중 오류가 발생했습니다: {e}")
        return None

# --- 모델의 '설계도' (Architecture) ---
class MultiTaskAST(nn.Module):
    """학습 때 정의했던 모델 구조를 그대로 가져옵니다."""
    def __init__(self, num_classes):
        super().__init__()
        # ASTModel의 파라미터(label_dim, input_tdim 등)는 학습 때와 동일하게 설정합니다.
        self.base_ast = ASTModel(label_dim=527, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False)
        self.classification_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
        self.regression_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))

    def get_embedding(self, x):
        x = x.unsqueeze(1); x = x.transpose(2, 3); B = x.shape[0]
        x = self.base_ast.v.patch_embed(x)
        cls_tokens = self.base_ast.v.cls_token.expand(B, -1, -1); dist_token = self.base_ast.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1); x = x + self.base_ast.v.pos_embed; x = self.base_ast.v.pos_drop(x)
        for blk in self.base_ast.v.blocks: x = blk(x)
        x = self.base_ast.v.norm(x); x = (x[:, 0] + x[:, 1]) / 2
        return x

    def forward(self, x):
        embedding = self.get_embedding(x)
        class_output = self.classification_head(embedding)
        decibel_output = self.regression_head(embedding)
        return class_output, decibel_output

print("예측에 필요한 함수와 모델 구조가 정의되었습니다.")


# ================================================================= #
# 단계 3: 모델 로딩 및 예측 준비
# ================================================================= #
#  ******** 사용자 설정 영역 ********
# 학습 때 사용했던 클래스 목록과 순서가 정확히 일치해야 합니다.
CLASSES = ['desk', 'chair', 'lecturestand', 'book', 'hammer']
# 학습 후 생성된 'best_model.pth' 파일의 경로를 지정해주세요.
MODEL_PATH = 'best_model.pth' # Colab 현재 세션에 파일이 있다면 이대로 사용
# 만약 모델이 구글 드라이브에 있다면 아래 코드의 주석을 해제하여 사용하세요.
# from google.colab import drive
# drive.mount('/content/drive')
# MODEL_PATH = "/content/drive/MyDrive/경로/best_model.pth"
# ***********************************

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(CLASSES)
idx_to_class = {i: name for i, name in enumerate(CLASSES)}

# 1. 모델의 '뼈대'를 생성합니다.
model = MultiTaskAST(num_classes=num_classes)

# 2. 'best_model.pth' 파일에서 학습된 가중치를 불러와 뼈대에 채워 넣습니다.
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"'{MODEL_PATH}'에서 학습된 가중치를 성공적으로 불러왔습니다.")
else:
    raise FileNotFoundError(f"모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. 경로를 확인해주세요.")

# 3. 모델을 GPU 또는 CPU로 이동시키고, '추론 모드'로 변경합니다.
model = model.to(device)
model.eval() # 예측 시에는 반드시 eval() 모드를 사용해야 합니다.

print(f"모델이 {device} 장치에 로드되었으며, 예측 준비가 완료되었습니다.")


# ================================================================= #
# 단계 4: 새로운 오디오 파일 예측 실행
# ================================================================= #

def predict_single_audio(file_path, model_to_use, device_to_use):
    """하나의 오디오 파일을 받아 모델로 예측하고 결과를 반환하는 함수"""
    print(f"\n▶ '{os.path.basename(file_path)}' 파일에 대한 예측을 시작합니다...")

    # 1. 오디오 파일을 학습 때와 동일한 형태로 전처리합니다.
    features = make_features(file_path, mel_bins=128)
    if features is None:
        return "오류", "파일 처리 실패"

    # 2. 모델에 입력하기 위해 차원을 추가하고(배치 차원), 디바이스로 보냅니다.
    #    [1024, 128] -> [1, 1024, 128]
    features = features.unsqueeze(0).to(device_to_use)

    # 3. 그래디언트 계산을 중지하여 메모리 사용량을 줄이고 속도를 높입니다.
    with torch.no_grad():
        # 모델 예측
        class_preds, decibel_preds = model_to_use(features)

        # 4. 모델의 출력(Output)을 해석하기 쉬운 형태로 변환합니다.
        # 분류 결과 처리
        predicted_idx = torch.argmax(class_preds, dim=1).item()
        predicted_label = idx_to_class[predicted_idx]

        # 데시벨 예측 결과 처리
        predicted_decibel = decibel_preds.squeeze().item()

    return predicted_label, predicted_decibel


# --- 실행 부분 ---
if __name__ == "__main__":
    # 예측하고 싶은 오디오 파일의 경로를 여기에 입력하세요.
    # 만약 파일이 구글 드라이브에 있다면 주석을 해제하여 경로를 지정하세요.
    # from google.colab import drive
    # drive.mount('/content/drive')
    TEST_AUDIO_FILE = "/content/drive/MyDrive/AI 학습/test_audio_files/test_chair_001.wav"

    if os.path.exists(TEST_AUDIO_FILE):
        # 예측 함수 호출
        label, decibel = predict_single_audio(TEST_AUDIO_FILE, model, device)

        # 결과 출력
        print("\n--- 최종 예측 결과 ---")
        print(f"입력 파일: {os.path.basename(TEST_AUDIO_FILE)}")
        print(f"예측된 소음원: {label}")
        print(f"예측된 데시벨: {decibel:.2f} dB")
        print("--------------------")
    else:
        print(f"\n[오류] 예측할 오디오 파일('{TEST_AUDIO_FILE}')을 찾을 수 없습니다. 경로를 다시 확인해주세요.")