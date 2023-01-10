
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0 \
python run.py \
--config /home/yichen/PCS-FUDA/config/mr2ct.json



outdim 512->64->16(memory bank)
bs 64->2

unlabel -> none transform

weight 怎么用？

debug-> img size=32
skip load-fewshot-to-cls-weight