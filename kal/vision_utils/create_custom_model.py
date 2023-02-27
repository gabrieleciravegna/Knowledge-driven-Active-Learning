import os.path


def create_custom_model_configuration_file(num_classes,
                                           folder,
                                           file_name,
                                           batch_size=12,
                                           lr=0.001,
                                           override=False):

    orig_dir = os.path.abspath(os.curdir)
    os.chdir(folder)

    if not os.path.exists(file_name) or override:

        with open(file_name, "w") as f:
            f.write(
                "\n"
                "[net]\n"
                "# Testing\n"
                "#batch=1\n"
                "#subdivisions=1\n"
                "# Training\n"
                f"batch={batch_size}\n"
                "subdivisions=1\n"
                "width=416\n"
                "height=416\n"
                "channels=3\n"
                "momentum=0.9\n"
                "decay=0.0005\n"
                "angle=0\n"
                "saturation = 1.5\n"
                "exposure = 1.5\n"
                "hue=.1\n"
                "\n"
                f"learning_rate={lr}\n"
                "burn_in=1000\n"
                "max_batches = 500200\n"
                "policy=steps\n"
                "steps=400000,450000\n"
                "scales=.1,.1\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=32\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "# Downsample\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=64\n"
                "size=3\n"
                "stride=2\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=32\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=64\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "# Downsample\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=3\n"
                "stride=2\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=64\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=64\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "# Downsample\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=2\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "# Downsample\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=2\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "# Downsample\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=1024\n"
                "size=3\n"
                "stride=2\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=1024\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=1024\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=1024\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=1024\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[shortcut]\n"
                "from=-3\n"
                "activation=linear\n"
                "\n"
                "######################\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=1024\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=1024\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=512\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=1024\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                f"filters={3 * (num_classes + 5)}\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[yolo]\n"
                "mask = 6,7,8\n"
                "anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\n"
                f"classes={num_classes}\n"
                "num=9\n"
                "jitter=.3\n"
                "ignore_thresh = .7\n"
                "truth_thresh = 1\n"
                "random=1\n"
                "\n"
                "\n"
                "[route]\n"
                "layers = -4\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[upsample]\n"
                "stride=2\n"
                "\n"
                "[route]\n"
                "layers = -1, 61\n"
                "\n"
                "\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=512\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=512\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=256\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=512\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                f"filters={3 * (num_classes + 5)}\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[yolo]\n"
                "mask = 3,4,5\n"
                "anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\n"
                f"classes={num_classes}\n"
                "num=9\n"
                "jitter=.3\n"
                "ignore_thresh = .7\n"
                "truth_thresh = 1\n"
                "random=1\n"
                "\n"
                "\n"
                "\n"
                "[route]\n"
                "layers = -4\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[upsample]\n"
                "stride=2\n"
                "\n"
                "[route]\n"
                "layers = -1, 36\n"
                "\n"
                "\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=256\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=256\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "filters=128\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "batch_normalize=1\n"
                "size=3\n"
                "stride=1\n"
                "pad=1\n"
                "filters=256\n"
                "activation=leaky\n"
                "\n"
                "[convolutional]\n"
                "size=1\n"
                "stride=1\n"
                "pad=1\n"
                f"filters={3 * (num_classes + 5)}\n"
                "activation=linear\n"
                "\n"
                "\n"
                "[yolo]\n"
                "mask = 0,1,2\n"
                "anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326\n"
                f"classes={num_classes}\n"
                "num=9\n"
                "jitter=.3\n"
                "ignore_thresh = .7\n"
                "truth_thresh = 1\n"
                "random=1"
                "\n"
                "\n"
            )

    os.chdir(orig_dir)


if __name__ == "__main__":
    create_custom_model_configuration_file(80, ".", "coco.cfg")
