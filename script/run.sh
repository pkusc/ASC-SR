mkdir LR/demo
mkdir LR/demo1
mkdir LR/tmp
cd RNAN_DN/code
CUDA_VISIBLE_DEVICES=0,3 python3 main.py --pre_train ../experiment/mya1/model/model_best.pt --save_results --test_only --model RNLRANSR --scale 4 --load ../experiment/mya1/ --testpath ../../LR/ --data_test Demo --n_GPU 2
cd ../../
python rename.py LR/demo
python test.py models/RRDB_ESRGAN_x4.pth
python inter.py LR/demo1 LR/demo LR/tmp
matlab -nodisplay -nosplash -nodesktop -r main_reverse_filter
