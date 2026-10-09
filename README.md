# 后室 · M.E.G. 回收录像（The Backrooms — Recovered Footage）

程序化生成的后室（The Backrooms）层级短片系列。画面、配乐和音效全部由代码实时合成：没有素材、贴图或 3D 模型，也不用 AI 生成画面。每集都是一卷被 M.E.G. 回收的录像，以档案卡开场，以档案卡结束，由同一位 M.E.G. 档案员旁白。

## 集数

| 集 | 层级 | 时长 | 1080p 成片 |
|---|---|---|---|
| 01 | [Level 254 —「蓝色天堂」Blue Heaven](episodes/ep01_level254/) | 3:25 | `out/ep01_level254/ep01_level254_1080p_web.mp4` |
| 02 | [Level 0 —「门槛」Threshold](episodes/ep02_level0/) | 3:25 | `out/ep02_level0/ep02_level0_1080p_web.mp4` |
| 03 | [Level 37 —「崇高」Sublimity（泳池房）](episodes/ep03_level37/) | 3:20 | `out/ep03_level37/ep03_level37_1080p_web.mp4` |
| 04 | [Level 881 —「窗之庭院」The Courtyard of Windows](episodes/ep04_level881/) | 3:18 | `out/ep04_level881/ep04_level881_1080p_web.mp4` |
| 05 | [Level 10 —「丰收」Bumper Crop](episodes/ep05_level10/) | 3:20 | `out/ep05_level10/ep05_level10_1080p_web.mp4` |
| 06 | [Level 11 —「不夜城」The City That Never Sleeps](episodes/ep06_level11/) | 3:20 | `out/ep06_level11/ep06_level11_1080p_web.mp4` |

## 渲染

```bash
pip install -r requirements.txt
episodes/<集名>/render_1080p.sh      # 1920x1080
episodes/<集名>/render_4k.sh         # 3840x2160（同一套代码，所有像素尺寸按分辨率缩放）
episodes/<集名>/remix_audio.sh       # 配音到位后重新混音并封装
python3 -m backrooms.render <集名> --still 30,90     # 只渲染几张静帧看效果
```

- 4 核 CPU 渲染 1080p 一集约 1 小时；4K 约 4 倍。可以加 `--jobs N` 用更多核。
- 输出都在 `out/<集名>/`。仓库里只提交 1080p 压缩版（`*_1080p_web.mp4`）。

## 配音

每集的 `voice/lines.json` 是配音台本，包含角色人设、语气指导、时间点和最长时长。交给本地 TTS 生成的方法见 [VOICE_AGENT_PROMPT.md](VOICE_AGENT_PROMPT.md)。人声文件放进 `episodes/<集名>/voice/wav/<id>.wav` 后，渲染时会自动混入：按时间点放置、音乐自动避让、按角色加效果（录音带、对讲机、远处回声等）。

## 结构

- `backrooms/core.py`：分辨率、摄像机、文字、VHS、故障、颗粒、HUD、字幕、档案卡
- `backrooms/sound.py`：合成器、乐器、滤波、混响
- `backrooms/gridworld.py`：网格世界逐像素光线投射器（薄墙、门洞、拱门、柱子、深坑、不同的地面和天花板高度、水面）
- `backrooms/city.py`：无尽城市的 2.5D 高度场光线投射器
- `backrooms/voice.py`：配音混入和音乐避让
- `backrooms/render.py`：渲染、混音、封装的命令行入口
- `episodes/epNN_xxx/`：每集的 `scene.py`（场景）、`film.py`（分镜）、`audio.py`（配乐）、`voice/lines.json`（配音台本）
