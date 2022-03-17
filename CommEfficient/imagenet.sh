#OMP_NUM_THREADS=8 python -m cProfile -o profile/cifar_fedsampler.pstats fed_train.py \
python3 cv_train.py \
    --dataset_dir /data/drothchild/datasets/imagenet/ \
    --local_batch_size 64 \
    --dataset_name ImageNet \
    --model FixupResNet50 \
    --local_momentum 0.0 \
    --virtual_momentum 0.9 \
    --weight_decay 1e-4 \
    --error_type virtual \
    --mode uncompressed \
    --iid \
    --num_clients 7 \
    --num_devices 8 \
    --num_workers 7 \
    --k 1000000 \
    --num_rows 1 \
    --num_cols 10000000 \
