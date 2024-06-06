export save_path='YOUR_SAVE_PATH'
export log_path='YOUR_LOG_PATH'

python eval.py --save_path ${save_path} --model resnet18 --modality image --datasets data_0528 \
| tee ${log_path}

# python eval.py --save_path ${save_path} --model resnet18 --modality text --datasets data_0528 \
# | tee ${log_path}

# python eval.py --save_path ${save_path} --model resnet18 --modality image+text --datasets data_0528 \
# | tee ${log_path}

# python eval.py --save_path ${save_path} --model resnet18 --modality image --datasets data_0528+data_0529 \
# | tee ${log_path}

# python eval.py --save_path ${save_path} --model resnet18 --modality text --datasets data_0528+data_0529 \
# | tee ${log_path}

# python eval.py --save_path ${save_path} --model resnet18 --modality image+text --datasets data_0528+data_0529 \
# | tee ${log_path}
