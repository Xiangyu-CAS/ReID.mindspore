使用mindspore和晟腾(Ascend)平台进行训练和推理的行人重识别(ReID)代码

## 支持功能
- [x] 将pytorch模型权重转换为mindspore(conv/pool pad对齐)
- [x] 基于mindspore和GPU训练/推理
- [x] 基于mindspore和Ascend训练/推理(FP16)
- [ ] 分布式训练
- [ ] 量化训练和推理(FP16, INT8)


## 环境要求
1. mindspore>=1.0.0, [INSTALL](https://www.mindspore.cn/install)
3. opencv-python, yacs


## checklist
- [x] baseline: resnet50 + random sampler + CrossEntropy Loss
- [x] FP16
- [ ] convert weights from pytorch
- [ ] pretrain model
- [ ] PK sampler
- [ ] dilation
- [ ] GeM
- [ ] BN1d
- [ ] circle loss
- [ ] pair-wise loss
- [ ] Distributed


## GPU 不支持的功能
- BN1d
- L2 Norm

