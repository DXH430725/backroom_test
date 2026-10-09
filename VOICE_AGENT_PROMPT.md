# 配音任务说明（交给本地 agent 执行）

把下面整段原样发给你本地（RTX 4090）的 agent。它会自己找出所有还没配音的集，逐集生成并推送。

---

```
任务：用本机 RTX 4090 部署开源中文 TTS，为一个短片系列批量生成配音，推送到 Git 分支。

## 仓库
- https://github.com/DXH430725/backroom_test
- 分支：claude/backrooms-level-254-video-6azv1g（只在这个分支工作）
- 开工前先 git pull，因为新的集数会不断被加进来。

## 要配的集
- 每一集是 episodes/<集名>/ 目录，台本在 episodes/<集名>/voice/lines.json。
- 需要配音的集：存在 voice/lines.json，但还没有 voice/wav/ 目录，或者 wav 数量少于 lines 数量。
- 已经配完的集不要重做，除非 lines.json 的修改时间晚于 wav。

## lines.json 结构
- speakers.<角色>.voice：音色和人设描述。同一集里不同角色必须是明显不同的声音。
- speakers.<角色>.series_voice：如果有，表示这个角色跨集出现（例如 "meg_archivist"），所有集里必须用同一个音色和参考音频。
- lines[]：id、speaker、tts_text（合成用的文本）、direction（语气和表演指导）、max_dur（秒）。
- 不要读 fx、pan、gain 字段，那是混音用的。你只交付干声。

## 模型
- 首选 IndexTTS2（index-tts/index-tts，支持情感控制和时长控制）。
- 备选：CosyVoice2（instruct 模式），然后是 F5-TTS。
- 用 conda 或 venv 隔离安装。全系列尽量用同一个模型，保证音色一致。

## 音色
- 用模型自带的预置说话人，或者许可证允许使用的开源数据集录音作为参考音频。不要克隆任何真实人物的声音。
- 每个角色的参考音频存到 voice_refs/<角色键>.wav（仓库根目录，所有集共享）。跨集角色用 series_voice 的值作为文件名。
- 在 voice_refs/README.md 里记录每个参考音频的来源和许可证。

## 生成要求（每一句）
1. 生成 3 个 take：episodes/<集名>/voice/takes/<id>_take{1,2,3}.wav
2. 格式：48kHz、单声道、24-bit WAV，干声（无混响、无音乐、无滤镜），峰值 ≤ -3 dBFS，首尾静音约 50ms。
3. 时长 ≤ max_dur。超了就用时长控制或语速重做，不要用变速拉伸（差 5% 以内可以例外）。
4. 按 direction 表演。"……"处要有明显停顿。
5. 质检：用 Whisper large-v3 或 Paraformer 转写，和 tts_text 比较（忽略标点）。有错字、漏字、吞字就重新生成。数字、英文和专有名词（如 "Rho 级"）重点检查。
6. 选出最好的一个 take，复制为 episodes/<集名>/voice/wav/<id>.wav

## 清单
每集写一个 episodes/<集名>/voice/manifest.json，内容包括：模型名和版本、每个角色的参考音频、每句选中的 take、时长、ASR 转写结果、是否超时。

## 提交
- 只提交 voice_refs/ 和各集的 voice/ 目录。不要改 .py 文件，不要提交 out/。
- 每完成一集提交一次，commit 信息写 "voice: <集名> (<模型名>)"。
- git push -u origin claude/backrooms-level-254-video-6azv1g
- push 被拒绝时（远端有新提交）：先 git pull --rebase，再推送。

## 汇报
每集汇报：用的模型、每句时长和 max_dur 的对比、重生成次数、最不满意的句子。
```

---

配音推上来之后，告诉 Claude「第 N 集配音好了」。混音会这样处理：
- 按时间点放入人声，人声说话时音乐自动压低。
- 按每个角色的 fx 加效果，例如访谈录音带、对讲机、远处回声。
- 重新封装成片。

你也可以自己在本地运行 `episodes/<集名>/remix_audio.sh`。
