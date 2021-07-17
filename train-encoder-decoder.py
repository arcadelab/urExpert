import tqdm
from torch import autograd
from dataset import KinematicsDataset
from utils import *
from model import *
from torch.utils.data import DataLoader
from datetime import datetime


if __name__ == "__main__":
    # user parameters
    # training specific
    suffix = "encoderdecoder-crossVid-selfVid-interval30" # -interval30
    num_epochs = 2000
    num_eval_epoch = 10
    lr = 0.0001
    weight_decay = 0.0001
    clip_max_norm = 2
    clip_norm_type = 2
    clip_value = 1e06
    save_ckpt = True
    is_penalize_all = True
    is_mask_enable = False
    is_gradient_clip = True
    is_debug = False

    # model specific
    load_checkpoint = False
    use_norm = False
    feat_dim = 512
    interm_dim = 2048
    nhead = 8
    num_decoder_layers = 6
    num_encoder_layer = 3
    input_channel = 6

    # dataset specific
    scale = 1000
    norm = False
    is_generalize = True
    is_extreme = False
    is_overfit = False
    is_zero_center = True
    is_gap = False
    drop_last = False
    input_frames = 150
    output_frames = 30
    interval_size = 30
    batch_size = 16 if is_generalize else 1 if is_extreme else 32

    scope = "general" if is_generalize else "overfit"
    task = "Needle_Passing"
    data_path = "./jigsaw_dataset_colab"
    task_folder = os.path.join(os.path.join(data_path, task), "kinematics")
    video_capture_folder = os.path.join(os.path.join(data_path, task), "video_captures")
    train_data_path = os.path.join(task_folder, "train") if is_generalize else os.path.join(task_folder,
                                                                                            "overfit_extreme" if is_extreme else "overfit")
    eval_data_path = os.path.join(task_folder, "eval") if is_generalize else os.path.join(task_folder,
                                                                                          "overfit_extreme" if is_extreme else "overfit")
    test_data_path = os.path.join(task_folder, "test") if is_generalize else os.path.join(task_folder,
                                                                                          "overfit_extreme" if is_extreme else "overfit")
    train_video_capture_path = os.path.join(video_capture_folder,
                                            "train" if is_generalize else "overfit_extreme" if is_extreme else "overfit")
    eval_video_capture_path = os.path.join(video_capture_folder,
                                           "eval" if is_generalize else "overfit_extreme" if is_extreme else "overfit")
    test_video_capture_path = os.path.join(video_capture_folder,
                                           "test" if is_generalize else "overfit_extreme" if is_extreme else "overfit")

    # encoder specific
    cross_attn_vid = True
    self_attn_vid = True
    is_feature_extract = False

    num_conv_layers = 3
    output_channel = 256
    conv_kernel_size = 3
    conv_stride = 1
    pool_kernel_size = 2
    pool_stride = 2
    padding = 1
    resize_img_height = 120
    resize_img_width = 160
    patch_height = 10
    patch_width = 10
    in_dim = 3
    capture_size = 2
    dropout = 0
    project_type = "conv"

    # create dataset
    train_dataset = KinematicsDataset(train_data_path, train_video_capture_path, input_frames=input_frames,
                                      output_frames=output_frames, is_zero_mean=False, scale=scale,
                                      is_zero_center=is_zero_center, is_overfit_extreme=is_extreme, is_gap=is_gap,
                                      interval_size=interval_size, norm=norm, capture_size=capture_size,
                                      resize_img_height=resize_img_height, resize_img_width=resize_img_width)
    eval_dataset = KinematicsDataset(eval_data_path, eval_video_capture_path, input_frames=input_frames,
                                     output_frames=output_frames, is_zero_mean=False, scale=scale,
                                     is_zero_center=is_zero_center, is_overfit_extreme=is_extreme, is_gap=is_gap,
                                     interval_size=interval_size, norm=norm, capture_size=capture_size,
                                     resize_img_height=resize_img_height, resize_img_width=resize_img_width)
    test_dataset = KinematicsDataset(test_data_path, test_video_capture_path, input_frames=input_frames,
                                     output_frames=output_frames, is_zero_mean=False, scale=scale,
                                     is_zero_center=is_zero_center, is_overfit_extreme=is_extreme, is_gap=is_gap,
                                     interval_size=interval_size, norm=norm, capture_size=capture_size,
                                     resize_img_height=resize_img_height, resize_img_width=resize_img_width)

    # create dataloaders
    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    loader_eval = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VideoEncoderDecoderTransformer(feat_dim=feat_dim, nhead=nhead, num_decoder_layers=num_decoder_layers,
                                           channel=input_channel, device=device, use_norm=use_norm,
                                           input_frames=input_frames, output_frames=output_frames, is_feature_extract=is_feature_extract,
                                           num_conv_layers=num_conv_layers, input_channel=input_channel,
                                           output_channel=output_channel, conv_kernel_size=conv_kernel_size,
                                           conv_stride=conv_stride, pool_kernel_size=pool_kernel_size,
                                           pool_stride=pool_stride, padding=padding, img_height=resize_img_height,
                                           img_width=resize_img_width, project_type=project_type,
                                           patch_height=patch_height, patch_width=patch_width, in_dim=in_dim,
                                           capture_size=capture_size, dropout=dropout, interm_dim=interm_dim,
                                           num_encoder_layer=num_encoder_layer, is_encode=self_attn_vid,
                                           attend_vid=cross_attn_vid)
    model.cuda()
    # check numbers of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # initialize loss function, optimizer, scheduler
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # load checkpoints if required
    if load_checkpoint:
        print("loading checkpoint...")
        time = "13_27"
        filename = time + suffix
        date_folder = os.path.join('checkpoints', sorted(os.listdir('checkpoints'))[-1])
        sub_folder = os.path.join(date_folder, filename)
        ckpt = torch.load(os.path.join(sub_folder, filename + ".ckpt"), map_location=None)
        model.load_state_dict(ckpt)
        print("loading checkpoint succeed!")
    else:
        # create checkpoint folder for saving plots and model ckpt
        if save_ckpt:
            now = datetime.now()
            now = (str(now).split('.')[0]).split(' ')
            date = now[0]
            time = now[1].split(':')[0] + '_' + now[1].split(':')[1] + suffix
            folder = os.path.join('./checkpoints', date)
            if not os.path.exists(folder):
                os.makedirs(folder)
            sub_folder = os.path.join(folder, time)
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

    # training starts
    train_loss = []
    eval_loss = []
    best_val_loss = 0
    for epoch in tqdm.tqdm(range(num_epochs)):
        if (epoch + 1) % num_eval_epoch != 0:
            print("Train Epoch {}".format(epoch + 1))
            loss_train = train(epoch, model, loader_train, optimizer, train_loss, device, loss_function, input_frames,
                               is_gradient_clip, clip_max_norm, clip_norm_type)
        else:
            print("Validation {}".format((epoch + 1) / num_eval_epoch))
            loss_eval = evaluate(epoch, model, loader_eval, scheduler, eval_loss, device, loss_function, input_frames)
            if best_val_loss == 0:
                best_val_loss = loss_eval
            if best_val_loss > loss_eval:
                best_val_loss = loss_eval
                print("New validation best, save model...")
                if save_ckpt:
                    save_model(time, sub_folder, model)
    if save_ckpt:
        plot_loss(sub_folder, train_loss, "train")
        plot_loss(sub_folder, eval_loss, "eval")
