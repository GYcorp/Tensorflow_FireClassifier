python Train_SingleGPU.py --use_gpu 0 --batch_size_per_gpu 32 --option b0 --augment randaugment --max_iteration 200000 --root_dir C:/DB/Recon_FireClassifier_DB_20200219/
python Train_SingleGPU.py --use_gpu 1 --batch_size_per_gpu 32 --option b0 --augment weakaugment --max_iteration 200000 --root_dir C:/DB/Recon_FireClassifier_DB_20200219/
python Train_SingleGPU.py --use_gpu 2 --batch_size_per_gpu 32 --option b0 --max_iteration 200000 --root_dir C:/DB/Recon_FireClassifier_DB_20200219/
