# LEVEL 254 —「蓝色天堂」Blue Heaven

后室（The Backrooms）Level 254 的程序化短片。约 3 分 25 秒，1920×1080，24 fps，立体声。

**成片：`out/level254_blue_heaven.mp4`**

片中所有东西都是代码生成的：没有素材、贴图或模型，也没有用 AI 生成画面。画面用 numpy 做光线投射和透视投影，用 OpenCV 做抗锯齿绘制，用 Pillow 排字；音乐和音效由 numpy/scipy 合成。

## 设定依据

依据 Backrooms Wiki（Wikidot）上的 Level 254「Blue Heaven」（已在 Mass Trimming Project 中归档）：

- 一片完全均匀的蓝色空间，没有建筑、地貌，也没有实体。
- 带有精神效应：进入的人会不惜一切想留下来，离开后这种效应也不会消退。
- **冥想圈（The Meditation Circle）**：受影响的流浪者终日静坐在地上，所有进入的人最终都会加入他们。
- M.E.G. 访谈记录 7350：唯一的已知生还者 Elise Beck 说那是"你这辈子见过的最美的蓝色，一直延伸下去"，"外面的世界已经不重要了"。她的队员还留在里面。
- 入口：从 Level 287 的墙壁 no-clip 进入。出口：从地板 no-clip 到 Level 11。
- 生存难度：Class ρ（不安全、不稳定、精神危害）。

Level 287（Glitched Halls）按其设定表现为：白墙、黑门、画面静电、逐渐褪成黑白、远处的钢琴声、戴帽子穿旧式西装的无脸人。

## 分镜

| 时间 | 段落 | 内容 |
|---|---|---|
| 0:00 | 档案卡 | M.E.G. 视听档案，打字机效果，警告"如感到平静请立即停止" |
| 0:11 | Level 287 | 白色走廊、黑门、闪烁的灯管、远处的钢琴华尔兹、走廊尽头的无脸人；画面逐渐褪成黑白，然后穿墙 |
| 0:29 | Level 254 | 纯粹的蓝。VHS 噪点和摄像机 HUD 慢慢"被治愈"并消失；片名 |
| 0:56 | +3 小时 | 地平线上出现一条细线；摄像机拉近变焦 |
| 1:14 | 冥想圈 | 最外圈的静坐者，绕着其中一人走半圈 |
| 1:36 | 延时 | 沿着放射状通道一路冲向圆心，两侧约 30 万人一闪而过 |
| 1:58 | 空位 | 最内圈留着一个空位。镜头走进去坐下，画面切成 2.39:1 宽银幕 |
| 2:18 | 离体上升 | 从座位直冲而上：刚才的空位上已经坐着一个人，就是"你"。俯瞰整个曼陀罗般的同心圆，绕行、远离，升入天空，合唱推到高潮 |
| 2:56 | 天堂 | 只剩蓝色 |
| 3:01 | 坠落 | 穿过地板 no-clip，瞥见一眼 Level 11（无尽之城），雨夜 |
| 3:07 | 结档 | 档案信息，受访者唯一的请求：「让我回去。」 |

## 重新生成

```bash
pip install numpy scipy pillow opencv-python-headless imageio-ffmpeg
python film/film.py --still 80,150        # 渲染几张静帧到 out/stills
python film/film.py --render              # 画面 → out/level254_video.mp4（4 核约 1 小时）
python film/audio.py                      # 配乐 → out/level254_audio.wav
python film/mux.py                        # 合成 → out/level254_blue_heaven.mp4
```

- `film/level254.py`：渲染器，包括高度雾天空、冥想圈（约 30 万个人物，分三级 LOD：胶囊体骨架、简化剪影、亚像素 splat）、软接触阴影、Level 287 走廊光线投射、Level 11 城市、VHS/故障/颗粒等后期
- `film/film.py`：分镜时间线、各镜头的摄像机运动、字幕、合成
- `film/audio.py`：配乐和音效，包括走调的钢琴华尔兹、日光灯嗡鸣、D 大调的 pad、共振峰滤波的合唱、钟声、心跳、呼吸声、雨声和卷积混响
