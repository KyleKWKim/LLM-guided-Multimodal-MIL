# commnets
: <<'END'
    Single GPU 든 Multi GPU 든 Batch size는 같아야 합니다! 그래야 정확한 비교가 됩니다!
    ( 대신 workers = (num_gpu) * workers )

    < Single Gpu Arguments >
    --gpu 0                                 # gpu number
    --dataset Task004_AVT_thickness1.5      # task name
    --model resunet                         # 
    --batch_size 4                          # 1~8
    --num_workers 8                         # 4 or 8
    --amp                                   # Automatic mixed precision     !!!!!!!!!!!!!왠만하면 amp 모드로 실행해 주세요!!!!!!!!!!
    --fold_num 0                            # fold number



    < Multi Gpu Arguments >
    --gpu 0,1,2,3                           # gpu numbers
    --dataset Task004_AVT_thickness1.5      # task name
    --model resunet                         # 
    --batch_size 4                          # 1~8
    --num_workers 32                        # 4 or 8   x  (num_gpu)
    --amp                                   # Automatic mixed precision  
    --fold_num 0                            # fold number
END



# export OPENBLAS_NUM_THREADS=1 # multiprocessing에서 2개 이상의 코어 사용하지 않게 하는 코드
# export OMP_NUM_THREADS=1

# python plot_TSNE.py --hospital_test [\'EUMC\',\'HUMC\',\'SCHMC\'] --tumorCrop 0 --clinical_features [\'sex\',\'age\',\'sm\',\'locationcancer\',\'cancerimaging\',\'cancerimagingT\',\'cancerimagingN\',\'cancerimagingM\',\'classification_cancer\'] --modality [\'pathology\'] --cancerstageTrain '1234' --cancerstageTest '1234' --model_CT 'resnetMC3_18' --gpu 0,1,2,3,4,5,6,7 --multiprocessing_distributed --lr 0.0001 --test_pth '/mnt/KW/LungCancer/Multimodality/results/SavedModels/EUMC+HUMC+SCHMC/modality(2)/stage_tr(1234)/norm_[2.0,2.0,2.5]/2024-01-11-11:59:42/checkpoint_0353.pth.tar'
# python plot_TSNE.py --hospital_test [\'EUMC\',\'HUMC\',\'SCHMC\'] --tumorCrop 0 --clinical_features [\'sex\',\'age\',\'sm\',\'locationcancer\',\'cancerimaging\',\'cancerimagingT\',\'cancerimagingN\',\'cancerimagingM\',\'classification_cancer\'] --modality [\'CI\'] --model_CI 'simpleFCs_v2' --cancerstageTrain '1234' --cancerstageTest '1234' --gpu 0,1,2,3,4,5,6,7 --multiprocessing_distributed --lr 0.0001 --test_pth '/mnt/KW/LungCancer/Multimodality/results/SavedModels/EUMC+HUMC+SCHMC/modality(3)/stage_tr(1234)/norm_[2.0,2.0,2.5]/2024-01-19-09:07:57/checkpoint_0756.pth.tar'
python plot_TSNE.py --hospital_test [\'EUMC\',\'HUMC\',\'SCHMC\'] --tumorCrop 0 --clinical_features [\'sex\',\'age\',\'sm\',\'locationcancer\',\'cancerimaging\',\'cancerimagingT\',\'cancerimagingN\',\'cancerimagingM\',\'classification_cancer\'] --modality [\'CT\',\'pathology\'] --batch_size 3 --gpu 4,5,6 --multiprocessing_distributed --test_pth '/mnt/KW/LungCancer/Multimodality2/results/SavedModels/EUMC+HUMC+SCHMC/modality(12)/stage_tr(1234)/resnetMC3_18-TransMIL(TransMIL)/norm_[0.6869,0.6869,3.0]/mask(X)/crop(X)/2024-03-03-15:03:36/checkpoint_best.pth.tar'