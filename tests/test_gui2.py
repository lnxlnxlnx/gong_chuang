import cv2
import numpy as np

def get_wrap_lines(text, font, font_scale, thickness, max_width):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_str = f"{current_line} {word}" if current_line else word
        (w, h), baseline = cv2.getTextSize(test_str, font, font_scale, thickness)
        if w <= max_width:
            current_line = test_str
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

# ========= 配置 =========
W, H = 800, 600
bg_color = (255, 255, 255)       # BGR 白色背景
txt_color = (0, 0, 0) # 黑色
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 4.2
thickness = 16
margin = 10  # 左右留白
text = "134 + 123+314 + 231"
# ========================

canvas = np.zeros((H, W, 3), dtype=np.uint8)
canvas[:] = bg_color
max_text_width = W - margin * 2
lines = get_wrap_lines(text, font, font_scale, thickness, max_text_width)

# 行间距
line_space = 120
# 整体垂直居中
(text_w0, text_h0), _ = cv2.getTextSize("Ay", font, font_scale, thickness)
total_height = len(lines) * line_space
y_start = (H - total_height) // 2

y = y_start
for line in lines:
    (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
    x = (W - tw) // 2
    cv2.putText(canvas, line, (x, y), font, font_scale, txt_color, thickness)
    y += line_space

cv2.imshow("Math", canvas)
print("按任意键关闭窗口")
cv2.waitKey(0)
cv2.destroyAllWindows()
