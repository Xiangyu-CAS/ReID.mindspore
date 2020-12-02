使用mindspore和晟腾(Ascend)平台进行训练和推理的行人重识别(ReID)代码

## 支持功能
- [ ] 基于mindspore和GPU训练/推理
- [ ] 基于mindspore和Ascend训练/推理(FP16)
- [ ] 分布式训练
- [ ] 量化训练和推理(FP16, INT8)


## 环境要求
1. mindspore>=1.0.0, [INSTALL](https://www.mindspore.cn/install)
3. opencv-python, yacs


## checklist
- [ ] baseline: resnet50 + random sampler + CrossEntropy Loss
- [ ] dilation
- [ ] GeM
- [ ] BN1d
- [ ] circle loss
- [ ] pair-wise loss
- [ ] Distributed


## GPU 不支持的功能
- BN1d
- L2 Norm
