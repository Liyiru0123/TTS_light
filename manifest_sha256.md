# 教师认可权重 SHA-256 校验清单

> 权重置于推理代码期望路径（`results/...`、`models/external/...`）；SHA-256 与 `P3-W05_model_integration/model_manifest.json` 逐项核对一致（测试 `PASS`）。`12位前缀` 用于课程材料中的标识。

| 权重路径 | SHA-256（前 12 位） |
|---|---|
| `results/P2-W03_tiny_vlm/checkpoints/tiny_vlm_inference.pt` | `8a10b5a1a889` |
| `results/P2-W04_tiny_image_ldm/revision_v2/training_attempt3_control/checkpoints/best.pt` | `4953bea21f20` |
| `results/P2-W05_tiny_asr/iterations/I002/branches/robust_finetune/inference_package.pt` | `6922c506a054` |
| `results/P3-W01_tiny_video_mllm/iterations/I003/branches/spacetime_queries/inference_package.pt` | `d8e7cb50b04f` |
| `results/P3-W03_tiny_tts/iterations/I004/branches/supported_rate/checkpoints/best.pt` | `82a297bc8f42` |
| `results/P3-W04_video_generation/iterations/I008/branches/generated_motion_module/checkpoints/best.pt` | `ddd4f0ad0681` |
| `results/P2-W02_image_vae/checkpoints/best.pt` | `aec680ec3aa9` |
| `results/P3-W02_speech_vae/iterations/I002/branches/hifigan_vae/inference_package.pt` | `7e6eb2c32c5c` |
| `models/external/hifigan_v3_ljspeech/hifigan_vocoder_v3_ljspeech.pth` | `4401d1af120a` |
| `results/P2-W01_training_framework/checkpoints/best.pt` | `ad4ca6a6e0ce` |
| `results/P2-W04_tiny_image_ldm/checkpoints/tiny_image_ldm_inference.pt` | `6bca9b8504c5` |
| `results/P2-W05_tiny_asr/iterations/I001/branches/baseline/checkpoints/best.pt` | `c1358b79ecd4` |
| `results/P3-W03_tiny_tts/checkpoints/best.pt` | `0a7ec52b1a8a` |
| `results/P3-W03_tiny_tts/iterations/I003/branches/rate_robust/checkpoints/best.pt` | `4f2790757bb2` |

> 后五行是复现链所需的**共享依赖/中间权重**（P2-W01 判别基线、LDM 推理包、ASR I001 基线、TTS 基础版与 I003 父权重），非六类“教学候选”，随包分发并同样受 SHA 校验。

> 身份注记：`technical_candidate`、`teaching_use_version`、`official_release` 为独立登记；六类模型 `official_release` 均为空，教学引用须写"教学候选（非正式发布）"，禁止称"最新版"。完整 64 位摘要见 `最小系统研发/results/P3-W05_model_integration/model_manifest.json`。
