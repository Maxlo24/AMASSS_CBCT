from monai.networks.nets import UNETR

def Create_UNETR(input_channel, label_nbr,cropSize):

    model = UNETR(
        in_channels=input_channel,
        out_channels=label_nbr,
        img_size=cropSize,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.05,
    )

    return model
