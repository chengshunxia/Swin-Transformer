python main_ipu.py 	\
       	--cfg configs/swin/swin_base_patch4_window7_224.yaml \
	--local_rank 0					     \
	--data generated				     \
	--iterations 1000				     \
	--batch-size 1					     \
	--run-ipu					     \
	--accumulation-steps 30
#       	--data-path /localdata/cn-customer-engineering/zhiweit/ai-datasets/datasets/imagenet/raw/imagenet-raw-data/  \
#	--data imagenet
