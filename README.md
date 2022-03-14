# CHIP2021 Task1 医学对话阴阳性判别

## 任务背景

比赛名称：医学对话临床发现阴阳性判别任务（CHIP2021评测一）

比赛结果：第2名（Macro-F1：77.87%）

方案思路：https://zhuanlan.zhihu.com/p/480244141

任务简介：针对互联网在线问诊记录中的临床发现的部分进行阴阳性的分类判别。 评测任务给定医患在线对话的完整记录，以及医患交互中提及的临床发现，要求对临床发现的阴阳性类别做判断。

## 数据集

本次评测任务的数据来源于春雨医生的互联网在线问诊公开数据。中文医疗信息处理挑战榜CBLUE公开了数据集下载，地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414

CHIP2021评测任务一的数据集可在CBLUE发布的CHIP-MDCFNPC任务中申请下载。CBLUE公开的8,000条数据包括5,000条训练数据、1,000条验证数据和2,000条测试数据。请将下载后的数据保存在`data/dataset`路径下。

原评测任务在CHIP2021中一共提供了9,999条数据，包括6,000条训练数据、2,000条A榜测试数据和1,999条B榜测试数据。本方案对该9,999条数据集进行五折划分，结果可在`data/dataset/split.csv`中查看，其中前6,000条为训练数据，6,000~8,000条为基于A榜测试数据（伪标签）的划分，最后1,999条为基于B榜测试数据（伪标签）的划分。比赛中原6,000条训练数据的后1,000条在MDCFNPC数据集中被公开为验证数据，前5,000条保留为训练数据，A榜测试数据已公开，B榜测试数据未公开。

## 环境依赖

- 主要基于 Python (3.7.3+) & AllenNLP 实现

- 实验使用 GPU 包括：Tesla V100 / TITAN RTX / GeForce GTX 1080Ti

- Python 版本依赖：


```
torch==1.8.1
transformers==4.5.1
allennlp==2.4.0
pandas==0.24.2
```

## Quick Start

#### 预训练模型

实验中选择了6种不同（规模）的开源预训练模型：

1. Chinese CPT-base，下载地址：https://huggingface.co/fnlp/cpt-base
2. RoBERTa-wwm-ext-base，下载地址：https://huggingface.co/hfl/chinese-roberta-wwm-ext
3. RoBERTa-wwm-ext-large，下载地址：https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
4. MacBERT-base，下载地址：https://huggingface.co/hfl/chinese-macbert-base
5. MacBERT-large，下载地址：https://huggingface.co/hfl/chinese-macbert-large
6. Chinese-bert-wwm，下载地址：https://huggingface.co/hfl/chinese-bert-wwm

请将下载后的模型权重`pytorch_model.bin`保存在`PLMs`路径下相应名称的模型文件夹中.

#### 数据预处理

```shell
python data_preprocess.py --input_file ./data/dataset/CHIP-MDCFNPC_train.jsonl --output_path ./data/dialog_data
```

- 按照`data/dataset/split.csv`中的数据划分进行五折交叉验证
- 公开数据集中的部分索引存在问题，`data/dataset/fix_badcase.jsonl`用来修正部分数据的索引，其余索引错误在代码中被修复
- 参数：{input_file}: 预处理的训练集路径，{output_path}: 预处理的输出文件夹

#### 模型训练

```shell
python trainer.py --train_file ./data/dialog_data/train/train0.pkl --dev_file ./data/dialog_data/valid/valid0.pkl --pretrained_model_dir ./PLMs/Roberta_base --output_model_dir ./save_model/Roberta_base/save_model_0 --cuda_id cuda:0 --batch_size 10 --num_train_epochs 5 --patience 2 --gradient_accumulation_steps 2
```

- 基于Chinese CPT-base模型时，使用文件名后缀为cpt，参考自：https://github.com/fastnlp/CPT
- 参数：{train_file}: 训练数据集路径，{dev_file}: 验证数据集路径，{pretrained_model_dir}: 预训练语言模型路径，{output_model_dir}: 模型保存路径

#### 模型预测

```shell
python predict.py --test_input_file ./data/dataset/CHIP-MDCFNPC_test.jsonl.txt --test_output_file ./prediction_results/Roberta_base/submission_0.txt --test_probs_file ./prediction_results/Roberta_base/probs_0.json --model_dir ./save_model/Roberta_base/save_model_0 --pretrained_model_dir ./PLMs/Roberta_base --cuda_id cuda:0 --batch_size 48
```

- 参数：{test_input_file}: 测试数据集路径，{test_output_file}: 预测结果输出路径，{test_probs_file}: 预测标签概率输出路径，{model_dir}: 加载的已训练模型路径，{pretrained_model_dir}: 预训练语言模型路径

## 如何引用

```
@Misc{WinningHealth2022,
      author={Yiwen Jiang},
      title={A Shared Embedding Strategy Based Model for Clinical Findings Classification in Medical Dialogues},
      year={2022},
	  howpublished={GitHub},
      url={https://github.com/winninghealth/chip/MDCFNPC},
}
```

## 版权

MIT License - 详见 [LICENSE](LICENSE)

