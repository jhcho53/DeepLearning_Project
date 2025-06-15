import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

# ———— 설정 ————
IMAGE_DIR = "/media/jaehyeon/T311/DeepLearning/Dataset/VOCdevkit/VOC2012/JPEGImages"            # 원본 RGB 이미지 폴더
MASK_DIR  = "/media/jaehyeon/T311/DeepLearning/Dataset/VOCdevkit/VOC2012/SegmentationClass"    # 새롭게 매핑된 7-클래스 마스크 폴더
OUTPUT_CSV = "voc2012_7class_dataset_analysis.csv"   # 결과 저장할 CSV

# 7개 클래스 이름
CLASSES = [
    "road",      # 0
    "sidewalk",  # 1
    "object",    # 2
    "vegetation",# 3
    "sky",       # 4
    "human",     # 5
    "vehicle",   # 6
]
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255

# ———— 데이터 파일 리스트 수집 ————
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
mask_paths  = sorted(glob.glob(os.path.join(MASK_DIR,  "*.png")))

assert len(image_paths) == len(mask_paths), "이미지·마스크 개수가 일치하지 않습니다!"

# ———— 통계 집계용 객체 ————
# (1) 이미지당 클래스 등장 횟수: mask에 해당 클래스가 한 픽셀이라도 있으면 +1
img_count_per_class   = Counter()
# (2) 전체 픽셀 수 집계
pixel_count_per_class = Counter()
# (3) 해상도 분포
size_counter = Counter()

# 각 파일별 메타정보를 담을 리스트
rows = []

for img_path, mask_path in zip(image_paths, mask_paths):
    # (A) 해상도 수집
    img = Image.open(img_path)
    w, h = img.size
    size_counter[(w, h)] += 1

    # (B) 마스크 읽고 라벨 통계
    mask = np.array(Image.open(mask_path))
    labels, counts = np.unique(mask, return_counts=True)

    # 이 이미지에 등장한 클래스들을 저장할 집합
    present = set()

    # 픽셀 단위 집계
    for lab, cnt in zip(labels, counts):
        if lab == IGNORE_INDEX:
            continue
        if 0 <= lab < NUM_CLASSES:
            pixel_count_per_class[lab] += int(cnt)
            present.add(int(lab))

    # 이미지 단위 집계
    for lab in present:
        img_count_per_class[lab] += 1

    # 행 정보 저장
    rows.append({
        "filename":        os.path.basename(img_path),
        "width":           w,
        "height":          h,
        "unique_labels":   [int(l) for l in labels if l != IGNORE_INDEX and 0 <= l < NUM_CLASSES],
        "pixel_counts":    {int(l): int(c) for l, c in zip(labels, counts) if 0 <= l < NUM_CLASSES},
    })

# ———— 판다스 DataFrame 생성 및 CSV 저장 ————
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[+] 파일별 메타정보를 '{OUTPUT_CSV}' 로 저장했습니다.\n")

# ———— 터미널 출력용 요약 통계 ————
print("=== 데이터셋 요약 ===")
print(f"총 이미지 수: {len(df)}\n")

print("-- 클래스별 이미지 등장 개수 --")
for i, cls in enumerate(CLASSES):
    print(f"{i:2d} {cls:10s}: {img_count_per_class[i]:5d}  images")

print("\n-- 클래스별 전체 픽셀 수 --")
for i, cls in enumerate(CLASSES):
    print(f"{i:2d} {cls:10s}: {pixel_count_per_class[i]:8d}  pixels")

print("\n-- 해상도 분포 (width × height : count) --")
for (w,h), cnt in size_counter.most_common():
    print(f"{w:4d}×{h:<4d}: {cnt:4d} images")

total_pixels = sum(pixel_count_per_class.values())
print("\n-- 클래스별 픽셀 점유율 (%) --")
for i, cls in enumerate(CLASSES):
    pct = 100.0 * pixel_count_per_class[i] / total_pixels if total_pixels > 0 else 0
    print(f"{i:2d} {cls:10s}: {pct:6.2f}%")
