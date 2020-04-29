python scripts/preprocess.py --data_dir data/mixamo/36_800_24/train -i 32 -s 64 -p 0
python scripts/preprocess.py --data_dir data/mixamo/36_800_24/test -i 60 -s 120 -p 0
python scripts/rotate_test_set.py --data_dir data/mixamo/36_800_24/test --out_dir data/mixamo/36_800_24/test_random_rotate
