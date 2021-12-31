class HyperParams:
    ################################
    # Experiment Parameters        #
    ################################
    epochs = 1000
    iters_per_checkpoint = 1000
    batch_size = 1

    n_gpus = 1
    dist_filepath = ''

    use_fp16 = False

    ################################
    # Data Parameters              #
    ################################
    output_directory = ''
    checkpoint_path = None
    load_model_state_dict_only = True

    train_inputs_path = ''
    train_targets_path = ''
    test_inputs_path = ''
    test_targets_path = ''

    train_identity = False

    ################################
    # Model Parameters             #
    ################################
    input_channels = 1
    filters = [64, 128, 256] # ResUNet filters
    n_D = 4 # number of downsampling discriminators
    n_df = 64 # number of discriminator filters in first layer

    ################################
    # Learning Parameters          #
    ################################
    lr = 1e-5
    beta1 = 0.5
    beta2 = 0.999
    epsilon = 1e-4
    lambda_FM = 10

    clip_grad = True
    clip_grad_thresh = 1.0

    lr_decay = True
    lr_decay_factor = 0.8
    lr_decay_interval = 500
