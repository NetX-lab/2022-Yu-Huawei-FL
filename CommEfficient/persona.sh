#OMP_NUM_THREADS=8 python -m cProfile -o profile/cifar_fedsampler.pstats fed_train.py \
python3 gpt2_train.py \
    --dataset_dir /home/yuzhang/CommEfficient/CommEfficient/datasets/persona/ \
    --local_batch_size 8 \
    --dataset_name PERSONA \
    --local_momentum 0.0 \
    --virtual_momentum 0.9 \
    --weight_decay 1e-4 \
    --error_type virtual \
    --mode uncompressed \
    --iid \
    --num_clients 7 \
    --device cpu\
    --num_devices 8 \
    --num_workers 7 \
    --num_results_train 1\
    --k 1000000 \
    --num_rows 1 \
    --num_cols 10000000 \
