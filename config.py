import os
current_dir = os.path.dirname(os.path.abspath(__file__))
args = {
    'data': {
        'dataset': 'PoLNS',
        # 'n_sub_classes': 4,
      #   'dataset': 'PoLEW',
      #   'n_sub_classes': 5,
      #   'data_dir': 'sate_data/train/',
      #   'data_dir': 'sate_data/train_with_timestamps/',
        'data_dir':  os.path.join(
            current_dir,  # 当前脚本所在目录
            '..',  # 返回上一级目录
            'ToHillstate',  # 进入同级的 "Create Mega Satellite" 文件夹
            'Result',  # 进入 "InPutData" 文件夹
            'div',
        ),
        'label_file': 'sate_data/train_labels.csv',
        'selected_features': ["x", "y", "z", "Vx", "Vy", "Vz"],
        'num_workers': True,
    },   
   'encoder_params':{
      "use_instance_norm":False, 
      "num_layers":9,
      "num_f_maps":64,
      "input_dim":6,
      "kernel_size":12,
      "normal_dropout_rate":0.3,
      "channel_dropout_rate":0.3,
      "temporal_dropout_rate":0.3,
      "feature_layer_indices":[1,2]
    },
   'decoder_params':{
      "num_layers":3,
      # "num_f_maps":64,
      "time_emb_dim":128,
      "kernel_size":13,
      "dropout_rate":0.1,
   },
   'diffusion_params':{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      # "cond_types":  ['full', 'zero','boundary03-', 'boundary05-', 'segment=1', 'segment=2'],
      "cond_types":  ['full', 'zero'],
     "detach_decoder": False,
   },
   "loss_weights":{
      # "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      # "encoder_boundary_loss":0.0,
      # "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      # "decoder_boundary_loss":0.01
   },
   'training': {
         "split_id":1,
         "sample_rate":1,
         "soft_label":None,
         "temporal_aug":True,
         "batch_size":4,
         "learning_rate":0.0005,
         "weight_decay":1e-5,
         "num_epochs":1000,
         "seed": 42,
         "device": "cuda:0",
    }
}


