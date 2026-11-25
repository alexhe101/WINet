# WINet — release-compatible modernization

Pinned reference: [original public commit 0101209](https://github.com/alexhe101/WINet/tree/0101209285d54ea8346715225e04a43b764b138b). This is an implementation refactor, not a paper-inspired redesign.

## Compatibility contract

- Separate wavelet transforms, lifting/HF enhancement, reconstruction and the network.
- Keep all LL/HL/LH/HH operations, stage order, bicubic interpolation with align_corners=True, active and inactive registered parameter names (including wop3).
- Replace hardcoded CUDA output allocation in IWT with input-device allocation, keeping the released FP32 output dtype. No wavelet library substitution or changed subband order.
- The old source, training code and evaluation scripts remain available.

The state dict names and shapes remain the same as the released `Net`. The strict loader handles bare weights or the original `net` wrapper and an all-keys DataParallel prefix. Unused registered tensors are deliberately retained. The repository does not contain public trained checkpoints. Tests use the actual released model's initialized weights, not recovered pretrained weights.

### Installation

Python 3.10+ and PyTorch 2.1+. Install a PyTorch build appropriate for your CPU or GPU. SciPy is required for MSDDN's original LU initialization and for original-reference tests.

```bash
python -m pip install -e './modern[dev]'
python -m pytest modern/tests -v -rs
```

## Input and output contract

Use one NPZ per co-registered scene, with finite floating CHW arrays in the same band order and radiometric range as the trained model. No implicit normalization, clipping, resize, quantization or image-mode conversion.

| Key | Shape | Required |
|:--|:--|:--|
| ms | [4,h,w] | Always |
| pan | [1,4h,4w] | Always |
| target | [4,4h,4w] | Supervised training |

For the released 4-band weights, the expected bands are RGB + near-infrared; preserve the exact original dataset channel order and normalization. Use FP32 for reference comparisons. Use CPU or CUDA, matching the model and input devices.  Same-sized scenes must be batched together. The output NPZ has an unbatched floating-point `hrms` array with all four bands.

```python
import torch
from winet import build_model, predict, load_weights

model = build_model(device="cpu")
load_weights(model, "your_checkpoint.pth")
model.eval()
ms = torch.rand(1,4,32,32,device="cpu")
pan = torch.rand(1,1,128,128,device="cpu")
with torch.inference_mode():
    hrms = predict(model, ms, pan)
```

Random inputs above only demonstrate the tensor contract. They are not satellite results.

```bash
winet infer --device cpu --checkpoint your_checkpoint.pth --input scene.npz --output result.npz
winet benchmark --device cpu --height 128 --image-width 128 --batch 1
winet train --device cpu --data paired_train --output runs/train
```

The new optional NPZ trainer keeps the active released default objective: L1 with mean reduction divided by 2× the configured batch size, Adam at 5e-4, gradient clipping 4, and CosineAnnealingWarmRestarts(T0=500, Tmult=1, eta_min=5e-8), stepped once per epoch. It is single-device and accepts prepaired data; original data degradation, augmentation, validation metrics, optimizer-resume behavior and exact training run are not reimplemented. The original solver/config are preserved for historical experiments. The modern `--checkpoint` argument initializes weights, not optimizer resume. Benchmarks synchronize CUDA and report synthetic latency and allocated memory, not claimed speedups. No AMP, compiler, alternate kernel or tiling is enabled silently.

## Validation

The suite compares the refactored model against the actual published source at width 4 and width 32: state-dict keys, strict loading, outputs, input gradients and every active parameter gradient. The CPU reference changes only the original IWT allocation's device; DWT/IWT roundtrip is checked separately. It also checks the CLI and one-step synthetic training. These are regression tests on synthetic inputs; they are not remeasurements of the paper's dataset quality.

[Automated test run](https://github.com/alexhe101/WINet/actions/runs/45547477493). Results must be interpreted according to the specific commit and test environment.

## Original code and publication

- [Original model](https://github.com/alexhe101/WINet/blob/0101209285d54ea8346715225e04a43b764b138b/model/winet.py)
- [Original configuration](https://github.com/alexhe101/WINet/blob/0101209285d54ea8346715225e04a43b764b138b/option.yml)
- [Publication](https://doi.org/10.1109/TGRS.2024.3367165)
