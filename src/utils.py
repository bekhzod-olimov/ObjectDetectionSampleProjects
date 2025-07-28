import os

def makedirs(dirname): os.makedirs(dirname, exist_ok=True)

def edit_custom_detection_yaml(
    yaml_path, 
    train_img_folder, train_ann_file, 
    val_img_folder, val_ann_file,
    num_classes=None
):
    import yaml
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['train_dataloader']['dataset']['img_folder'] = train_img_folder
    cfg['train_dataloader']['dataset']['ann_file'] = train_ann_file
    cfg['val_dataloader']['dataset']['img_folder'] = val_img_folder
    cfg['val_dataloader']['dataset']['ann_file'] = val_ann_file
    if num_classes is not None:
        cfg['num_classes'] = num_classes

    with open(yaml_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    print(f"custom_detection.yml updated with new paths!")


