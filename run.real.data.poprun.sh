poprun --mpi-global-args="--allow-run-as-root --tag-output" --num-instances=8 --numa-aware=yes --num-replicas=16 --ipus-per-replica=1 \
python main_ipu.py 	\
       	--cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
	--local_rank 0					     \
	--batch-size 4					     \
	--run-ipu					     \
       	--data-path /localdata/cn-customer-engineering/zhiweit/ai-datasets/datasets/imagenet/raw/imagenet-raw-data/  \
	--executable-cache-dir ./cache	\
	--data imagenet			\
	--accumulation-steps 128	\
	--replication-factor 2		\
	--dataloader-worker 4
