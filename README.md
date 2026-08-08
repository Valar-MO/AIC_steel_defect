# AIC Steel Defect Detection

九类高分辨率钢板表面缺陷目标检测项目。

## 当前阶段

1. 数据与标注体检
2. 固定、按采集序列分组的本地验证集
3. Zero-shot 与监督检测 baseline
4. 全图缩放和切图训练/推理对照
5. 官方 submission.json 生成与校验

## 数据安全

官方数据属于保密数据，不得提交至 Git 或上传到公开仓库。数据应存放在仓库外：

```text
/root/autodl-tmp/datasets/AIC_steel_defect/
```

代码仓库默认位置：

```text
/root/autodl-tmp/AIC_steel_defect/
```

任何可能包含官方图片、标注内容或样本文件名的实验产物，在公开前必须人工检查。

## 目录

```text
configs/       实验与类别配置
docs/          实验记录和数据报告
scripts/       数据检查、训练、评估、推理入口
src/           可复用 Python 模块
tests/         坐标变换和提交格式测试
```

