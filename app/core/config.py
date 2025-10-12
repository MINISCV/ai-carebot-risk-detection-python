import torch

# --- Configuration ---
LABEL_MODEL_PATH = "./model/label"
SUMMARY_MODEL_PATH = "./model/summary"
LABEL_ORDER = ['positive', 'danger', 'critical', 'emergency']
MAX_TOTAL_TEXT_LENGTH = 10000
EVIDENCE_COUNT = 2

# --- Device Setup ---
def get_device():
    """사용 가능한 디바이스(CUDA, MPS, CPU)를 확인하고 반환합니다."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()