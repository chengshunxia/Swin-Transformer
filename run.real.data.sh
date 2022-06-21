python main_ipu.py 	\
       	--cfg configs/swin/swin_base_patch4_window7_224.yaml \
	--local_rank 0					     \
	--batch-size 4					     \
	--run-ipu					     \
       	--data-path /localdata/cn-customer-engineering/zhiweit/ai-datasets/datasets/imagenet/raw/imagenet-raw-data/  \
	--executable-cache-dir ./cache	\
	--data imagenet			\
	--accumulation-steps 30 \
	--replication-factor 1
