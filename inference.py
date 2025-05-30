# inference_with_palette.py

import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# 7개 클래스용 팔레트: [R, G, B, R, G, B, …]
# 0=road(purple), 1=sidewalk(pink), 2=object(yellow), 3=vegetation(green),
# 4=sky(skyblue), 5=human(red), 6=vehicle(navy)
CUSTOM_PALETTE = [
    128,   0, 128,   # 0 road (purple)
    255, 192, 203,   # 1 sidewalk (pink)
    255, 255,   0,   # 2 object (yellow)
      0, 255,   0,   # 3 vegetation (green)
      0, 191, 255,   # 4 sky (sky blue)
    255,   0,   0,   # 5 human (red)
      0,   0, 128,   # 6 vehicle (navy)
]

# 모델 로드
def load_model(weights_path, device):
    model = deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=7)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# 전처리: 단일 이미지 → 배치 텐서
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = TF.resize(img, (512, 512), interpolation=Image.BILINEAR)
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return img.unsqueeze(0)

# 후처리: 로짓 → 클래스 인덱스 → PIL 팔레트 이미지
def postprocess(output, orig_size):
    pred = output.argmax(1).byte().cpu().numpy()[0]  # (H, W), 값 0~6
    mask = Image.fromarray(pred, mode='P')          # 모드 'P' = 팔레트
    mask.putpalette(CUSTOM_PALETTE)                  # 팔레트 적용
    # 원본 크기로 리사이즈 (팔레트 유지)
    mask = mask.resize(orig_size, resample=Image.NEAREST)
    return mask

# 추론 함수
def infer(model, input_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.[jp][pn]g")))
    else:
        files = [input_path]

    for fp in files:
        orig = Image.open(fp)
        w, h = orig.size
        x = preprocess(fp).to(device)
        with torch.no_grad():
            out = model(x)["out"]
        mask = postprocess(out, (w, h))

        fn = os.path.splitext(os.path.basename(fp))[0]
        save_path = os.path.join(output_dir, fn + "_color.png")
        mask.save(save_path)
        print(f"Saved colored mask: {save_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    type=str, required=True,  help="학습된 .pth 파일 경로")
    p.add_argument("--input",      type=str, required=True,  help="이미지 파일 혹은 디렉터리")
    p.add_argument("--output_dir", type=str, default="./masks", help="컬러 마스크 저장 폴더")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = load_model(args.weights, dev)
    infer(mdl, args.input, args.output_dir, dev)
