import matplotlib.pyplot as plt

# 클래스 이름, 퍼센티지, 그리고 매핑할 색상(hex 코드)
classes = ["Road", "Sidewalk", "Object", "Vegetation", "Sky", "Human", "Vehicle"]
percentages = [8.47, 2.37, 22.07, 42.63, 20.51, 1.06, 2.89]
colors = [
    "#800080",  # road (purple)
    "#FFC0CB",  # sidewalk (pink)
    "#FFFF00",  # object (yellow)
    "#00FF00",  # vegetation (green)
    "#00BFFF",  # sky (sky blue)
    "#FF0000",  # human (red)
    "#000080",  # vehicle (navy)
]

plt.figure(figsize=(8, 4))

# 수평 막대 그래프
plt.barh(classes, percentages, color=colors)

# 축 레이블과 타이틀 (폰트 크기 지정)
plt.xlabel("Pixel Percentage (%)", fontsize=24)
plt.title("Pixel share by class", fontsize=24)

# 틱 레이블 크기 조정
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# x축 범위
plt.xlim(0, max(percentages) + 5)

# 각 바 우측에 퍼센티지 값 표시 (폰트 크기 지정)
for idx, val in enumerate(percentages):
    plt.text(val + 0.3, idx, f"{val:.2f}%", va='center', fontsize=24)

plt.tight_layout()
plt.show()