########################################################
# Train
########################################################

# commands for windows
python Train_SingleGPU.py --use_gpu 0 --batch_size_per_gpu 32 --option b0 --augment randaugment --max_iteration 200000 --root_dir C:/DB/Recon_FireClassifier_DB_20200219/

python Train_SingleGPU.py --use_gpu 0 --batch_size_per_gpu 32 --option b0 --augment randaugment --max_iteration 200000 --root_dir C:/DB/Recon_FireClassifier_DB_20200219/ --valid_iteration 1000

# commands for linux
python3 Train_MultipleGPU.py --use_gpu 0,1,2,3,4,5,6,7 --batch_size_per_gpu 64 --option b0 --max_iteration 200000
python3 Train_MultipleGPU.py --use_gpu 0,1,2,3,4,5,6,7 --batch_size_per_gpu 64 --option b0 --augment randaugment --max_iteration 200000

