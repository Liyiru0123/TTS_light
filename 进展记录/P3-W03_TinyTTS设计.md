# P3-W03 TinyTTS 设计

> 工作包：`P3-W03 TinyTTS`
> 实现名：`TinySpeechFlow`
> 状态：已验收
> 授权依据：教师 2026年8月18日确认路线冻结
> 依赖：P1-W05（语音数据+对齐）、P2-W05（TinyASR）、P3-W02（SpeechVAE+HiFi-GAN）
> 开放台账：`DATA-SPG-001`（短时文本—语音训练单元）

## 1. 教学目标与实现边界

### 1.1 核心教学现象

本工作包要证明：**离散短文本如何变成变长连续语音**。具体链条：

$$\text{Text} \rightarrow \text{Alignment/Duration} \rightarrow \text{Frame-wise Fusion} \rightarrow \text{Conditional Flow} \rightarrow \text{Speech Latent} \rightarrow \text{Frozen Acoustic Decoder}$$

学生应理解：
- 文本长度 $N_{\text{text}}$ 与语音长度 $T_{\text{speech}}$ 不同，duration/alignment 解决这个失配；
- Cross-Attention 是"correspondence 不确定时的 soft 机制"，Duration 是"correspondence 有强结构时的 hard 机制"；
- 压缩不仅减少数据量，也可能损失对齐分辨率（representation resolution → alignment granularity）；
- Flow Matching 是概率运输的另一种建模视角，不是扩散的"升级版"。

### 1.2 必须实现

| 组件 | 说明 |
|---|---|
| Tiny Text Encoder | 双向 Transformer，上下文建模 |
| Duration Predictor | $h_i \rightarrow \log(d_i+1)$，仅 acoustic token |
| Length Regulator | $N_{\text{text}} \rightarrow T_{\text{latent}}$，repeat/插值 |
| Frame-wise Additive Conditioning | $y_t = x_t + W h_t$，共享 W，输入端一次注入 |
| 1D Temporal U-Net | CFM 主干，沿 T 轴 Conv1D，per-time channel normalization |
| Linear CFM | 概率运输，$x_0 \sim N(0,I)$，$x_t=(1-t)x_0+tx_1$，$v_\theta \approx x_1-x_0$ |
| 冻结 P3-W02 Decoder | 64 帧容器，posterior mean 解码 |
| 冻结 HiFi-GAN V3 | 波形生成 |
| 冻结 TinyASR | 回译评价 |

### 1.3 不实现（边界）

| 不实现 | 原因 |
|---|---|
| GAN / 对抗训练 | CPU 不可行，课程无先例 |
| VAE 重训 | P3-W02 已验收，冻结 |
| 多说话人 / 情感 / 韵律 | 单声数据，one-to-many 讲概念不验收 |
| DiT | N30 映射，不上主干 |
| 完整 OT-CFM | 称 linear CFM，诚实命名 |
| Cross-Attention 主干 | 研发候选，第一版不实现 |
| MAS / 学习对齐 | N30 映射 |
| LM / speech token | N30 映射 |
| world model | N30 延伸 |
| one-to-many prosody 验收 | 单 realization 数据，CFM 随机性≠韵律多样性 |

## 2. 信息流

### 2.1 训练路径（training-only）

```
真实语音
  ↓
冻结 P3-W02 编码器
  ↓
posterior mean μ [4, 20, 64]  ← 训练 target x_1
  ↓
Linear CFM 训练:
  x_0 ~ N(0,I) [4, 20, 64]
  x_t = (1-t)·x_0 + t·x_1
  v_θ(x_t, t, c) ≈ x_1 - x_0
  loss = MSE(v_θ, v)  (只在 valid mask 处)
```

**注意**：posterior mean 是训练 target 的来源，不是推理路径上的模块。

### 2.2 推理路径

```
文本 (字符+标点)
  ↓
embedding + 位置
  ↓
Tiny Text Encoder (双向 Transformer)
  ↓
contextualized tokens [N, D]
  ↓
Duration Predictor  h_i → log(d_i+1)
  推理: d̂_i = max(0, round(exp(log(d̂_i+1))-1))
       T̂ = Σ d̂_i  ← 由预测决定输出长度，不参考 T_valid
  ↓
Length Regulator  (N → T̂, repeat)
  ↓
Frame-wise Condition  H ∈ R^{D×T̂}
  ↓
Linear CFM (1D temporal U-Net, 含 timestep t 注入)
  x_0 ~ N(0,I) [4, 20, T_max=64]
  ODE 积分 → x(1) = ẑ [4, 20, 64]   ← 完整 64 容器, latent 不 crop
  ↓
冻结 P3-W02 decoder (始终输入完整 64 帧容器)
  ↓
80维 Log-Mel [80, 256]
  ↓
crop 到前 4·T̂ mel 帧   ← 在 mel 域裁剪, 不在 latent 域
  ↓
冻结 HiFi-GAN V3
  ↓
波形 (22.05 kHz)
  ↓
冻结 TinyASR
  ↓
CER / 时长误差 / 条件一致性 / 教师试听(PED)
```

## 3. 接口定义（10 项 + 3 项实证补充）

### 3.1 duration-bearing token 定义

- **包括**：汉字、标点（P1 数据中标点中位 64 mel 帧，99.8% > 0）
- **排除**：BOS、EOS、PAD
- **v1 约定**：$d_i^{\min} = 0$（允许零时长）
- **依据**：P1 中 4.8% token 的 mel duration ≤ 2 frames，说明 zero-duration 风险不可忽略
- **最终**：$P(d_i=0)$ 由 DATA-SPG-001 在 latent-center projection 后实测确定

### 3.2 Mel→latent alignment projection policy

本项是一个**人为但固定、单调、可复现的 projection convention**（在 DATA-SPG-001 中落实为算法），**不是**"求真实 receptive-field ownership"。

**c_j 的可执行定义（冻结为 nominal-bin convention，路线 A）**：P3-W02 编码器时间轴为两级 stride-2 下采样（256→128→64），**时间轴压缩比为 $4$**（相邻 latent 帧在 mel 时间轴上间隔 **4 帧**，不是 16）。

> 注意：latent 频率维 $80\to20$ 同样是 4×，但那是**频率轴**压缩，与 duration alignment 无关；不得把面积压缩 $4\times4=16$ 误用于时间轴。

latent 帧 $j$（$0\le j < 64$）在投影约定中被指派一个 **nominal 4-frame temporal bin** $[4j,\,4(j+1))$（时间压缩比 4，$64\times4=256$ 与输入 mel 帧数一致）。**该 bin 是人为投影约定，不是卷积 encoder 的真实 receptive field**（后者通常远大于 4 mel 帧）。

$$c_j = 4j + 2 \quad\text{（mel 帧坐标，冻结，不随 kernel/padding 改变）}$$

**边界校验**：$j=63 \Rightarrow c_{63}=4\times63+2=254 < 256$，全部 64 个 latent 中心都落在输入 mel 范围 $[0,256)$ 内（旧公式 $c_{63}=1016$ 已超出，证明 16 为错）。

**只遍历有效 latent centers（关键）**：设 $L_{\text{mel}}=b_N$（最后一个 token 的有效 mel 边界）。有效 latent 帧集合：

$$J_{\text{valid}} = \{\, j \mid 0\le j < 64,\ c_j < L_{\text{mel}}\,\}, \qquad T_{\text{valid}} = |J_{\text{valid}}|$$

**不得固定遍历 64**：$T_{\text{valid}}<64$ 时，$j\ge T_{\text{valid}}$ 的 latent 帧属 container padding，其 nominal bin 落入 padding，不能分给 token。

**归属**：token $i$（mel 边界 $[b_{i-1}, b_i)$）拥有 latent 帧 $j$ 当且仅当 $c_j \in [b_{i-1}, b_i)$。

**时长（集合基数）**：

$$d_i = \left|\{\, j \in J_{\text{valid}} : c_j \in [b_{i-1}, b_i)\,\}\right|$$

**伪代码**：
```
L_mel = b_N                                  # 最后一个 token 的有效 mel 边界
valid_js = [ j for j in range(64) if (4*j + 2) < L_mel ]
assert len(valid_js) == T_valid_from_data    # ★ Gate A 一致性检查

d = [0]*N
for j in valid_js:
    c_j = 4*j + 2                            # nominal 4-frame bin 中心
    i  = token whose [b_{i-1},b_i) contains c_j   # 单调, 唯一
    d[i] += 1
assert sum(d) == len(valid_js) == T_valid_from_data
```

- **性质**：deterministic、monotonic、每个有效 latent 帧恰属一个 token、$\sum_i d_i = T_{\text{valid}}$
- **诚实承认**：$d_i=0$ 是表示压缩分辨率不足（一个 token 区间内无 latent 帧中心落入），不是 alignment 出错
- **命名**：称"alignment projection convention"，不是 ground-truth latent alignment
- **教学**：mel alignment 是 ground truth（真值）；latent alignment 是由压缩表示诱导的 deterministic projection
- **Gate A 一致性检查**：$|J_{\text{valid}}|$ 必须等于 P1 的 $T_{\text{valid}}$；若不等，说明两套 valid-length 定义不一致，**在 Gate A 暴露并禁止进入 Gate B**，不能偷偷继续训练
- **零时长率统计**：$P(d_i=0)$ 只在 $J_{\text{valid}}$ 上统计，不受 padding 帧污染

### 3.3 T_valid、T_hat、T_max 严格区分

| 符号 | 含义 | 来源 |
|---|---|---|
| $T_{\text{valid}}$ | 该单元的有效 latent 帧数 | P1 对齐决定 |
| $\hat{T}$ | predicted duration 的总和 | 推理时由模型决定 |
| $T_{\max} = 64$ | 最大窗口（batch 对齐用） | 教学系统约定 |

**教学**：64 是当前教学系统的最大窗口，不是 TTS 的固有输出长度。

### 3.4 T_hat > T_max 显式失败策略

- TinyTTS v1 支持范围：$1 \le \hat{T} \le 64$
- 若 $\hat{T} > 64$：显式报告 `duration_overflow`，该文本超出当前教学模型支持范围
- 不自动压缩、不静默截断
- **教学**：真实 TTS 为什么需要 chunking、streaming、long-form generation

### 3.5 Mask 进入 CFM state / velocity / loss / 每个 temporal block

- 初始化：$x_0 \leftarrow x_0 \odot M$
- 网络输入：mask 掉 padding
- **mask 粒度（关键）**：$M$ 必须在**任何下一次 temporal-mixing Conv1d 之前**、且**在前一次运算可能使 invalid 位置重新非零之后**重新施加。
  - 原因：一个 residual block = `2×Conv1d + 2×LayerNorm + time add`。第一层 Conv 后 invalid 位置 $h_L$ 可能因 Conv bias、$h_{L-1}$ 邻域、time-embedding 而变非零；若等整个 two-conv block 做完才 mask，第二次 Conv 的 kernel=3 已把这些 invalid 值卷回 valid 边界。
  - 契约：**若前一次运算可能使 invalid 位置非零，则在下一次 temporal-mixing Conv1d 之前必须重新 $\odot M$**。
  - 参考粒度（block 内）：`Conv → LN → time add → act → **mask** → Conv → LN → act → **mask** → residual add → **mask**`
  - 下采样到 $T=32$ 尺度时，用对应前缀 mask（valid 长度 $\lceil T_{\text{valid}}/2\rceil$）
- velocity：$v_\theta \leftarrow v_\theta \odot M$
- ODE 每步：$x_{t+\Delta t} \leftarrow (x_t + \Delta t \cdot v_\theta) \odot M$
- Loss：只在 $M=1$ 处计算
- 说明：mask 不只是"最后裁掉 padding"，而是真正进入生成过程；粒度按上条契约，保证 invalid activation 不进入下一次 temporal Conv

### 3.6 Training-only posterior-mean target path 与 inference path 分图

- **训练**：真实语音 → 冻结编码器 → $\mu$ → $x_1$ target
- **推理**：Text → Duration → Condition → CFM → $\hat{z}$ → 冻结解码器 → Speech
- **注意**：posterior mean 不是 CFM 后的模块，是训练 target 的来源

### 3.7 Frame-wise condition：shape 链与注入位置（闭合）

Length Regulator 输出 $H_{\hat{T}} \in \mathbb{R}^{\hat{T} \times D_{\text{text}}}$（$\hat{T}=T_{\hat{}}$ 个 frame，每帧文本维 $D_{\text{text}}=64$）。投影、补齐到 64 容器、mask，再一次性加到输入：

$$H_{\hat{T}} \xrightarrow{\,W\,} C_{\hat{T}} \in \mathbb{R}^{80 \times \hat{T}} \xrightarrow{\,\text{PadTo64}\,} C_{64} \in \mathbb{R}^{80 \times 64} \xrightarrow{\,\odot M\,} C_{64}, \qquad \boxed{y = x_t + C_{64}}$$

- $W: D_{\text{text}} \rightarrow 80$（latent channels），**逐帧共享** Linear，参数 $64 \times 80 + 80 = 5{,}200$（独立于 U-Net 参数，不计入 4.5 的 684,240）
- `PadTo64`：在时间维尾部 zero-pad 到 64（$\hat{T}<64$ 时）；$\hat{T}>64$ 由 Length Regulator 报 `duration_overflow`
- `C_{64} \odot M`：padding 帧置零，mask 与 CFM 用同一前缀 mask
- **注入位置**：只在 stem 之前一次（$y=x_t+C_{64}$）；U-Net 各 temporal block **只注入 timestep $t$**，不重复注入 text condition
- 参数量：$T$ 影响计算量，不影响参数量
- 说明：与 4.3 结构图、第 10 节接口契约一致；`y_t=x_t+W h_t` 的逐帧写法等价于 $y=x_t+C_{64}$ 的批量写法

### 3.8 PF 按 Gate A → B → C 三级组织

- **Gate A**：P3-W02 回归 + decoder 变长兼容 + zero-latent-duration 统计 + padding-value compatibility
- **Gate B**：ground-truth duration CFM 过拟合 + Text ON/OFF/shuffled
- **Gate C**：Predicted duration

### 3.9 Text ON/OFF/shuffled 消融

| 条件 | 说明 |
|---|---|
| A. Correct text + correct ground-truth duration | 基准 |
| B. Zero text condition + same ground-truth duration | 无文本内容 |
| C. Shuffled text condition + same ground-truth duration | 文本语义错乱 |

- **结论**：在相同时长规划下，linguistic frame condition 对生成内容具有显著贡献
- **注意**：ground-truth duration 本身也包含由文本带来的 timing information

### 3.10 Duration × rate 属教学现象

- duration ×0.7 / ×1.0 / ×1.3
- 展示"Duration controls speaking rate"听觉现象
- 不属正确性验收（改变 duration 后 latent/Mel reconstruction 指标无直接意义）

### 3.11 [实证补充] d_i=0 允许

- P1 中 4.8% token 的 mel duration ≤ 2 frames
- v1 不应预设 $d_{\min}=1$
- 最终 zero-duration rate 由 projection 后统计决定

### 3.12 [实证补充] latent-center projection

- 不试图从 convolution receptive field 求"真实 token ownership"
- 称"alignment projection convention"
- mel alignment 是 ground truth（真值）；latent alignment 是由压缩表示诱导的 deterministic projection

### 3.13 [实证补充] 64 帧容器 + per-time norm

- P3-W02 decoder accepts variable T but is not length-invariant because of GroupNorm
- P3-W03 fixes decoder input container to T=64
- 变长通过 valid masks/cropping 表示，不改变 decoder temporal shape
- CFM 生成器用 per-time channel normalization（LayerNorm/RMSNorm over channels）
- **对照**：旧模块为什么必须 containerize，新模块为什么主动避免长度耦合

## 4. 1D temporal U-Net 结构（速度场网络 v_θ）

### 4.1 输入/输出

- 输入：$x_t \in [B, 80, T_{\max}=64]$（80 = 4×20，latent channels × frequency）、**timestep $t \in [0,1]$**、frame-wise condition $c$
- 输出：velocity 场 $v \in [B, 80, 64]$
- **注意**：$v_\theta$ 依赖 $(x_t, t, c)$ 三者，$t$ 不可省略——即使 linear path 下 target $x_1-x_0$ 固定，条件速度场 $\mathbb{E}[x_1-x_0\mid x_t,t,c]$ 一般仍依赖 $t$

### 4.2 Time conditioning（timestep 注入）——必须实现

```
t (scalar)
  ↓
e = MLP([t, sin(πt), cos(πt)])          # e ∈ R^{D_e=128}
  ↓  每个 temporal block 内:
     W_block·e  (Linear, D_e → block_channels)
  ↓
加到该 block 的 channel feature (broadcast over T)
```

- $D_e=128$；time MLP = `Linear(3→128)+GELU+Linear(128→128)`
- 每个 temporal block 有一个 `Linear(D_e → C_block)` 投影，投影结果 broadcast 到时间维加到特征上
- **契约**：代码实现者不得写出 $v_\theta(x_t,c)$（缺 $t$）；那将不是本设计定义的 CFM

### 4.3 结构与时间形状（冻结）

```
x_t [B,80,64]
  + C_64 [B,80,64]          ← text condition, 输入端一次注入 (见 3.7)
  = y [B,80,64]
    ↓
stem Conv1d(80→128,k3,p1)        → [B,128,64]
block0         (time)            → [B,128,64]   T=64
down0 Conv1d(128→192,k3,s2,p1)   → [B,192,32]   T=32  (stride-2 下采样)
block1         (time)            → [B,192,32]   T=32
up0  Conv1d(192→128,k3,p1)+skip  → [B,128,64]   T=64  (nearest 2× 上采样)
block2         (time)            → [B,128,64]   T=64
out  Conv1d(128→80,k1)           → [B,80,64]
```

- **条件注入分工**：
  - **text condition**：只在输入端一次（$y=x_t+C_{64}$），之后不再重复注入
  - **timestep $t$**：每个 temporal block 都注入（4.2 的 time-proj），因为 $v_\theta$ 依赖 $t$
- **时间形状**：$64 \rightarrow 32 \rightarrow 64$（真 U-Net，含 skip）
- **下采样**：`Conv1d(stride=2)`
- **上采样**：`nearest` 2× 插值 + `Conv1d`（不用 ConvTranspose，避免 checkerboard）
- **skip**：`down0` 前的 `[B,128,64]` 与 `up0` 输出相加
- **卷积**：kernel 3，padding 1（沿 T 轴）

### 4.4 Normalization 与 mask（每个 temporal block）

- **Normalization**：per-time channel（LayerNorm over channels），**不用 GroupNorm**（避免跨 T 统计 → 长度耦合）
- **Mask 粒度**：$M$ 必须在**任何下一次 temporal-mixing Conv1d 之前**、且在前一次运算可能使 invalid 位置非零之后**重新施加**（不是"等整个 two-conv block 做完才 mask"——那样第二次 Conv 已把 invalid 值卷回 valid 边界）
  - 原因：一个 residual block 含 `2×Conv1d`；第一层 Conv 后 invalid 位置可能因 bias/邻域/time-embedding 变非零，须在被第二次 Conv 读取前复位
  - 参考：`Conv → LN → time add → act → **mask** → Conv → LN → act → **mask** → add → **mask**`
  - 2× 下采样后，$T=32$ 尺度用对应前缀 mask（valid 前缀长度 $\lceil T_{\text{valid}}/2\rceil$）
  - valid 区域始终是前缀 $[0, T_{\text{valid}})$，用前缀 mask，不需要任意 mask

### 4.5 参数估算（完整 block 定义，含 time embedding / time-proj / LayerNorm）

| 组件 | 参数 |
|---|---:|
| time MLP  `Linear(3→128)+Linear(128→128)` | 17,024 |
| stem `Conv1d(80→128,k3)` | 30,848 |
| block0 `time-proj(128→128)+2×Conv1d(128)+2×LN(128)` | 115,584 |
| down0 `Conv1d(128→192,k3,s2)` | 73,920 |
| block1 `time-proj(128→192)+2×Conv1d(192)+2×LN(192)` | 247,104 |
| up0 `Conv1d(192→128,k3)+add skip` | 73,856 |
| block2 `time-proj(128→128)+2×Conv1d(128)+2×LN(128)` | 115,584 |
| out `Conv1d(128→80,k1)` | 10,320 |
| **总计** | **684,240** |

- 此估算**已含** time embedding、逐 block time-projection、LayerNorm
- frame-wise condition $W: D_{\text{text}}\to 80$（$64\times80+80=5{,}200$）独立计入，不在此表
- CPU 可行性由 PF 资源测试确认并留机器证据（非预设结论）

## 5. 风险表

| 风险 | 类型 | 说明 |
|---|---|---|
| duration/mask 接口 | 最高接口风险 | mel→latent 映射、off-by-one、padding、0帧、超T_max |
| CFM convergence / intelligibility | 最高学习风险 | frozen latent 分布、conditioning、ODE steps、manifold |
| frozen decoder compatibility | 最高系统接口风险 | GroupNorm 跨 T 统计、64 帧容器 |

## 6. PF 预检清单

### Gate A：P3-W02 回归 + 接口检查

1. P3-W02 回归测试（不训练 CFM）：speech → μ → Frozen Decoder → speech，验证与 P3-W02 验收结果一致
2. decoder 变长兼容性测试：T=16/32/48/64 能否正常解码
3. zero-latent-duration 统计：$\frac{\#{d_i^{\text{latent}}=0}}{\#\text{tokens}}$
4. padding-value compatibility check：zero latent padding vs P3-W02 原训练数据 padding
   - **失败处置**：若 zero-padding 导致 valid-region decoder output 相对 reference padding 出现不可接受偏差，则 **Gate A fail，禁止进入 Gate B**；必须先冻结与 P3-W02 decoder 训练分布一致的 padding policy
   - 说明：Gate A 的职责就是允许架构在这里被否决——这才是真正的 gate
5. mel→latent alignment projection 正确性：$\sum_i d_i = T_{\text{valid}}$

### Gate B：ground-truth duration CFM

1. ground-truth duration + CFM 过拟合（极小数据）
2. generated latent 可解码
3. ASR 内容正确性
4. Text ON/OFF/shuffled 消融

### Gate C：Predicted duration

1. Predicted duration vs ground-truth duration
2. CER 差
3. 诊断树：
   - Gate A 失败 → P3-W02 接口问题
   - Gate A 好, B 差 → CFM / conditioning 问题
   - Gate B 好, C 差 → duration planning 问题
   - 全部好 → TinyTTS 闭环成立

## 7. 教学边界

| 边界 | 说明 |
|---|---|
| one-to-many prosody | 讲概念，不验收 |
| 64 帧 | 最大窗口，非固有长度 |
| linear CFM | 非完整 OT-CFM |
| 单声 | 不宣称韵律多样性 |
| 时长先验 | 可检验，非韵律 |

## 8. 对齐/融合课程一级主线

### 8.1 核心问题

$$\boxed{\text{Alignment: Who corresponds to whom?}}$$
$$\boxed{\text{Fusion: Once aligned, how is information combined?}}$$

### 8.2 跨模态对齐/融合表

| 场景 | Alignment | Fusion |
|---|---|---|
| 图文检索 | shared embedding space | similarity |
| VQA / 文生图 | Cross-Attention | Cross-Attention |
| ASR | CTC latent paths | acoustic→text decoding |
| **TTS** | **Duration + Length Regulator** | **frame-wise additive condition** |
| VideoQA | temporal attention | multimodal fusion |
| 文生视频 | text conditioning + temporal modeling | temporal interaction |

### 8.3 Cross-Attention 的特殊性

- Cross-Attention = Alignment + Fusion 一体化
- TinyTTS = Alignment first（Duration）→ Fusion second（frame-wise additive）
- **教学**：Attention 强大，不代表 Attention 是多模态的定义

### 8.4 N29 统一

- 表征、对齐、融合、条件生成、时间建模的共性
- 跨模态迁移
- "How do two representations correspond?"

## 9. 不做清单

- GAN / 对抗训练
- VAE 重训
- 多说话人 / 情感 / 韵律
- DiT
- 完整 OT-CFM
- Cross-Attention 主干
- MAS / 学习对齐
- LM / speech token
- world model
- one-to-many prosody 验收

## 10. 接口契约表

| 组件 | 输入 | 输出 | Shape | Mask | 错误处理 |
|---|---|---|---|---|---|
| Text Encoder | 字符+标点 | contextualized tokens | [N, D] | — | N=0 报错 |
| Duration Predictor | contextualized tokens | log(d_i+1) | [N, 1] | — | N=0 报错 |
| Length Regulator | tokens + durations | frame-level condition | [T̂, D] | — | T̂>64 报 duration_overflow |
| Frame-wise Condition | $H_{\hat T}$ [T̂, D] | $C_{64}$ (pad+mask) | [80, 64] | $\odot M$ | T̂>64 由 Length Regulator 报 duration_overflow |
| 1D Temporal U-Net | $y=x_t+C_{64}$, **t (逐 block timestep)** | velocity v | [B, 80, 64] | M(每个 temporal block 后) | — |
| Linear CFM | x_0, v_θ(·,t,c) | x(1) = ẑ [4,20,64] 完整容器 | [B, 4, 20, 64] | M | — |
| P3-W02 Decoder | ẑ [4,20,64] 完整 64 容器 | mel | [B, 1, 80, 256] | — | T̂=0 报错 |
| Mel Crop | mel [80,256] | cropped mel | [B, 1, 80, 4·T̂] | — | — |
| HiFi-GAN | cropped mel | waveform | [B, 1, samples] | — | — |
| TinyASR | waveform | text | — | — | — |

## 11. 主要证据路径

| 类型 | 路径 |
|---|---|
| 数据 | `data/speech/p1_w05_v1/` |
| 配置 | `configs/p3_w03_tiny_tts.json` |
| 结果 | `results/P3-W03_tiny_tts/` |
| 设计 | `docs/P3-W03_TinyTTS设计.md` |
| 测试计划 | `docs/P3-W03_测试计划.md` |
| 指标说明 | `docs/P3-W03_指标与术语说明.md` |
| 模型卡 | `docs/P3-W03_TinyTTS模型卡.md` |

## 12. 验收项编号

| 编号 | 说明 |
|---|---|
| DATA-SPG-001 | 短时文本—语音训练单元 |
| MODEL-SPG-001 | CFM 生成器 |
| MODEL-SPG-002 | Duration Predictor |
| MODEL-SPG-003 | Length Regulator |
| PED-SPG-001 | 教师试听 |
| RES-SPG-001 | CPU 资源 |
| REG-SPG-001 | 回归测试 |

## 13. I001—I004技术设计演化

以下是正式登记迭代的设计轨迹，不把2026年8月21日未登记、未保留检查点的时长增强开发试跑冒充正式`I002`。

| 迭代 | 父迭代 | 要解决的问题 | 技术/结构变化 | 明确保留项 | 设计结论 |
|---|---|---|---|---|---|
| I001/main | 无 | 建立文本→时长→帧条件→linear CFM→冻结解码链 | 首次实现Text Encoder、Duration Predictor、Length Regulator、帧级加性条件、1D U-Net速度场和20步ODE | 80维SpeechVAE潜空间与HiFi-GAN冻结 | 形成770,465参数基线并通过原冻结机器验收 |
| I002/rate_robust | I001 | 修复倍率缩放造成的丢字/重字、padding泄漏与CFM路径问题 | 源码方案加入精确总长最大余数缩放、零时长保持、逐Token潜变量重采样、padding噪声隔离和CFM正向路径修正；拟做0.7—1.3时长增强 | 网络层数、通道、文本/时长/CFM主体和冻结解码链不变 | 配置快照缺`feature`块，训练前中止；0次参数更新，不能称为新模型结果 |
| I003/rate_robust | I001 | 在完整配置上验证上述修复与宽倍率增强 | 补全80维特征契约；从I001以lr 2e-4微调3轮；50%样本采用0.7—1.3精确时长增强 | 模型拓扑和参数量不变；只改实现契约与训练分布 | 产生检查点，但0.7倍快速内容门失败；只作为I004父检查点 |
| I004/supported_rate | I003 | 用较实际范围换取更稳定、可辨的语速控制 | 增强范围收窄至0.8—1.2，lr降为1e-4，只收敛1轮；推理仍用同一精确时长缩放 | 模型拓扑、CFM目标、冻结声学链不变 | 原ASR门失败而历史中止；后续客观配对和身份盲听支持其作为教学使用版本，不改变技术候选I001 |

设计层面的关键结论是：I002— I004没有新增网络模块，变化主要发生在时长缩放算法、mask/CFM实现正确性、训练倍率分布和支持范围。详细结果与证据见`docs/P3-W03_总结报告.md`及`results/P3-W03_tiny_tts/iteration_registry.json`。
