python main_ipu.py 	\
       	--cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
	--local_rank 0					     \
	--batch-size 4					     \
	--run-ipu					     \
       	--data-path /localdata/cn-customer-engineering/zhiweit/ai-datasets/datasets/imagenet/raw/imagenet-raw-data/  \
	--data imagenet
