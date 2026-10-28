import matplotlib.pyplot as plt

'''
ssh 屏幕显示:
export DISPLAY=:0
xrandr --output DP-1 --rotate normal
'''

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 匹配你屏幕 800x600
fig = plt.figure(figsize=(800/100, 600/100), dpi=100)
ax = fig.add_axes([0,0,1,1]) # [left,bottom,w,h] 占满整个画布，不留白
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.axis("off")

text_content = "134 + 123 + 314 + 231"
ax.text(0.5, 0.5, text_content, fontsize=70, ha='center', va='center', color="#000000")

plt.show()
input("继续")
