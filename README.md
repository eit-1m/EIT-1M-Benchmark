# EIT-1M-Benchmark
The source code for the benchmark experiments of EIT-1M dataset.

### Training

Simply run the training scripts as followed:

```shell
bash run.sh
```

### Testing

```shell
python eval.py --save_path MODEL_PATH --model resnet18 --modality image --datasets data_0528
```
