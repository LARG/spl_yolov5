conda activate D:\Github\yolov5\conda39
$ENV:PYTHONPATH="."
# python 'scripts/labels_from_cvat.py' 
# python 'scripts/labels_from_cvat_bottom.py' 
python 'train.py' '--cfg=./models/yolov5nnn7-top-yuv.yaml' '--data=data/benchmarks-multi.yaml' '--epochs=100' '--patience=300' '--img=384' '--batch=4' '--workers=1' '--cache'  '--hyp=data/hyps/hyp.bench.yaml' --weights=runs\train\exp436/weights/best.pt
python 'train.py' '--cfg=./models/yolov5nnn6-bot-yuv.yaml' '--data=data/benchmarks-multi-bot.yaml' '--epochs=100' '--patience=300' '--img=96' '--batch=16' '--workers=1' '--cache'  '--hyp=data/hyps/hyp.bench.bot.yaml'  --weights=runs\train\exp441\weights/best.pt