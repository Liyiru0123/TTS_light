# P3-W03 TinyTTS 模型卡

> 工作包：`P3-W03 TinyTTS`
> 实现名：`TinySpeechFlow`
> 模型卡版本：1.0
> 建立日期：2026年8月20日
> 依据：`docs/P3-W03_TinyTTS设计.md`（2026年8月18日冻结）

## 1. 教学任务与课程目标

**任务**：文本 → 变长连续语音（TTS）。

**课程目标**：
1. 理解离散短文本如何变成变长连续语音，核心是 $N_{\text{text}} \neq T_{\text{speech}}$ 的长度与对齐失配；
2. 区分 soft correspondence（Cross-Attention）与 hard/structured correspondence（Duration/Length Regulator）；
3. 理解压缩如何改变信息关系（Representation Resolution → Alignment Granularity）；
4. 接触 flow matching 作为概率运输的另一种生成视角（非扩散升级版）；
5. 理解多模态系统是"接口系统"（Representation + Transformation + Interface Contract）。

**标志性教学现象**：
- 关闭文本条件仍能生成语音，但内容错乱（条件贡献）；
- 时长 ×0.7/×1.0/×1.3 改变语速（Duration controls speaking rate）；
- 零时长 token 是压缩分辨率损失，不是对齐错误；
- overflow（$\hat{T}>64$）显式失败，引出真实 TTS 的 chunking/streaming。

## 2. 必须实现的结构

| 组件 | 结构 | 参数 |
|---|---|---:|
| Tiny Text Encoder | 双向 Transformer，D_text=64，embedding+位置 | 见 §4 |
| Duration Predictor | $h_i \to \log(d_i+1)$，仅 acoustic token | 见 §4 |
| Length Regulator | $N \to \hat{T}$，repeat | 0（无参数） |
| Frame-wise Condition | $W: 64\to80$，$y=x_t+C_{64}$ | 5,200 |
| 1D Temporal U-Net | 64→32→64，per-time channel norm，time embedding D_e=128 | 684,240 |
| Linear CFM | $x_t=(1-t)x_0+tx_1$，$v_\theta \approx x_1-x_0$ | 含在 U-Net |
| 冻结 P3-W02 Decoder | 64 帧容器，posterior mean 解码 | 冻结 |
| 冻结 HiFi-GAN V3 | 波形生成 | 冻结（1,462,273） |
| 冻结 TinyASR | 回译评价 | 冻结 |

## 3. 允许缩小或替换的结构

- 文本编码器层数/头数（D_text=64 固定）；
- U-Net 通道数（684,240 为估算，实际以代码为准）；
- 时长预测器深度；
- ODE 采样步数（教学用固定步数）。

## 4. 参数量估算

| 组件 | 参数 |
|---|---:|
| Text Encoder（双向 Transformer，D=64，2 层） | 约 40,000—60,000 |
| Duration Predictor | 约 5,000 |
| Frame-wise Condition W | 5,200 |
| 1D Temporal U-Net（含 time embedding） | 684,240 |
| **可训练合计** | **约 730,000—750,000** |
| 冻结 VAE（P3-W02） | 187,177 |
| 冻结 HiFi-GAN V3 | 1,462,273 |
| 冻结 TinyASR | 约 400,000 |

可训练参数 < 1,000,000，满足 `RES-SPG-001`。

## 5. 截至调研日期的代表性模型家族与高频技术

（调研日期：2026年8月18日）

| 真实系统 | 本模型映射 | 差距 |
|---|---|---|
| VITS / FastSpeech 2 | Duration Predictor + Length Regulator + 声学模型 | 本模型用 CFM 替代声学模型，无显式 pitch/energy |
| NaturalSpeech / AudioLDM | 潜空间生成 + 声码器 | 本模型用 linear CFM 替代扩散，无 CLAP 条件 |
| Flow matching (Lipman et al.) | velocity field + ODE 采样 | 本模型用 linear path，非 OT-CFM |
| HiFi-GAN | 冻结声码器 | 本模型冻结使用，不训练 |

## 6. 微型结构与真实结构的一一映射

| 本模型 | 真实系统对应 | 说明 |
|---|---|---|
| Text Encoder | 文本编码器（Transformer） | 同 |
| Duration Predictor | 时长预测器（FastSpeech/VITS） | 同 |
| Length Regulator | 长度调节器 | 同 |
| Frame-wise Condition | 帧级条件注入 | 真实系统常用 Cross-Attention，本模型用加性条件 |
| 1D Temporal U-Net | 声学模型/去噪网络 | 本模型用 1D U-Net，真实系统常用 Transformer |
| Linear CFM | 扩散/流匹配 | 本模型用 linear CFM |
| 冻结 VAE Decoder | 声谱图解码器 | 同 |
| 冻结 HiFi-GAN | 神经声码器 | 同 |

## 7. 简化项、原因和教学后果

| 简化项 | 原因 | 教学后果 |
|---|---|---|
| 无显式 pitch/energy | 单声数据，韵律不验收 | 不展示韵律多样性 |
| linear CFM（非 OT-CFM） | CPU 可行，诚实命名 | 不宣称概率运输最优 |
| 加性条件（非 Cross-Attention 主干） | 第一版不实现 | 展示 alignment-first→fusion-second 路线 |
| 64 帧容器 | 教学系统约定 | 不展示长语音 chunking/streaming |
| 单声 | 数据限制 | 不展示多说话人 |

## 8. 资源预算

- 设备：CPU，8 线程
- 可训练参数：< 1,000,000
- 训练墙钟：≤ 3,600 秒
- 峰值 RSS：≤ 2.5 GiB

## 9. 验收项编号和验收结果

| 验收项编号 | 说明 | 状态 |
|---|---|---|
| `DATA-SPG-001` | 短时文本—语音训练单元 | 已派生（8,968训练单元，验收全过） |
| `MODEL-SPG-001` | CFM 生成器 | I001 已训练（Gate B 过拟合 0.835；冻结链基准差距 −0.0134） |
| `MODEL-SPG-002` | Duration Predictor | I001 已训练（Gate C 时长相对误差 0.0247） |
| `MODEL-SPG-003` | Length Regulator | I001 已训练（0 溢出） |
| `PED-SPG-001` | 教师试听 | ✅ 已确认（2026-08-21；能力样例册：`results/P3-W03_tiny_tts/examples/capability_examples.md`） |
| `RES-SPG-001` | CPU 资源 | I001 训练 176s（预算 3600s） |
| `REG-SPG-001` | 回归测试 | Gate A 全过（P3-W02 回归、变长、padding、alignment） |

I001 详细结果见 `docs/P3-W03_总结报告.md`，验收决定见 `docs/P3-W03_教学验收决定.md`。I001 检查点归档于 `results/P3-W03_tiny_tts/iterations/I001_archive/`。I002 时长增强实验未采纳（见总结报告 §4）。教学风险见 `docs/教学使用风险台账.md` TEACH-TTS-001~005。

## 10. 学生自学与扩展方向

- 完整 OT-CFM 与 conditional flow matching；
- Cross-Attention 主干 vs 加性条件的对比；
- 显式 pitch/energy 建模（FastSpeech 2 / VITS）；
- 长语音 chunking/streaming；
- 多说话人与韵律多样性。

## 11. 2026-08-22语速严格复核

复核发现旧语速演示存在三项实现/验收缺陷：逐token四舍五入使相邻倍率总长相同；零时长标点被错误强制为1帧；ODE填充噪声在mask前进入卷积。现已改为“保持零时长＋最大余数精确总长”，并在ODE和Temporal U-Net的每个混合入口先应用mask。旧DTW丢字判据因完整路径必然覆盖参考帧而失效，替换为冻结ASR的字符插入/删除/替换统计。

I002因训练前配置校验失败且没有发生更新，按治理规则中止；I003的0.7×压力门失败；I004在预冻结的0.8–1.2范围仍出现8条样例下删除门非单调翻转，故其机器执行状态为中止。可靠通过项是时长控制契约和语速差异：0.8/1.0/1.2实际时长比约0.800/1.000/1.200。由于冻结ASR绝对CER约0.95–1.20、样本仅8条，“无丢字/多字”的自动证据不稳定。2026-08-22教师依据新增样例试听，决定以I004/supported_rate作为教学验收使用版本；自动门禁失败保留为评价边界，不再阻塞教学使用，且不得把该决定写成正式发布。

## 12. 模型结构与训练路线谱系

| 迭代 | 网络结构相对上一轮 | 数据/训练目标相对上一轮 | 当前身份与原因 |
|---|---|---|---|
| I001/main | 建立Text Encoder＋Duration Predictor＋Length Regulator＋帧条件＋1D U-Net CFM | 真值时长训练，无倍率增强；12轮，lr 1e-3 | 技术候选；通过原冻结机器验收 |
| I002/rate_robust | 拓扑不变；拟修复精确时长、mask和CFM实现 | 拟加入0.7—1.3逐Token时长增强 | 训练前配置失败，0次更新，不是可用模型 |
| I003/rate_robust | 拓扑和参数量不变；采用已修复实现 | 从I001微调3轮，50%概率0.7—1.3增强 | 压力门失败，中止；保留为父检查点与失败证据 |
| I004/supported_rate | 拓扑和参数量不变 | 从I003再微调1轮，增强范围收窄为0.8—1.2 | 教学使用；正常/快速客观误差较低且盲听清晰度3:0，但变速整体显著改善和内容完整非劣未证明 |

因此，I004不是“更大或增加了新模块”的模型，而是同一TinySpeechFlow在修正后的时长/mask实现和受控倍率训练分布上的版本。技术候选、教学使用和正式发布分别为`I001/main`、`I004/supported_rate`和`null`。
