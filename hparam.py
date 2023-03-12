class hparams:

    train_or_test = 'train'
    output_dir = '/home_data/teacher01/llx/segment-result'
    #是否进行数据增强
    aug = 'None'
    latest_checkpoint_file = 'checkpoint_latest.pt'
    #一共跑多少轮
    total_epochs = 100
    epochs_per_checkpoint = 10
    batch_size = 2
    ckpt = None
    init_lr = 0.002
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 5

    crop_or_pad_size = 512,512,128 # if 2D: 256,256,1
    patch_size = 128,128,32 # if 2D: 128,128,1

    # for test
    patch_overlap = 4,4,4 # if 2D: 4,4,0

    fold_arch = '*.nii'

    save_arch = '.nii'

    source_train_dir = '/home_data/teacher01/llx/segment-data/train/nii/data'
    label_train_dir = '/home_data/teacher01/llx/segment-data/train/nii/label'
    source_test_dir = 'test/image'
    label_test_dir = 'test/label'


    output_dir_test = 'results/your_program_name'