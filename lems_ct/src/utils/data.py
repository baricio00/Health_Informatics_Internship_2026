import pandas as pd
from pathlib import Path

def get_files_from_csv(input_dir, csv_path, fold_idx):
    input_dir = Path(input_dir) / "DATA_REGISTERED_FIXED_LATE"
    print(input_dir)
    print(csv_path)
    df = pd.read_csv(csv_path)
    print(df)
    
    train_files = []
    val_files = []
    
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        current_fold = int(row["fold"])
        
        # Dynamically build the absolute path using the Azure mount directory
        img_path = input_dir / patient_id / "CT_LATE.nii.gz"
        lbl_path = input_dir / patient_id / "registration_mask.nii.gz"
        
        if img_path.exists() and lbl_path.exists():
            data_dict = {"image": str(img_path), "label": str(lbl_path)}
            
            # Route to Train or Val based on the fold argument
            if current_fold == fold_idx:
                val_files.append(data_dict)
            else:
                train_files.append(data_dict)
                
    print(f"Loaded {len(train_files)} training scans and {len(val_files)} validation scans for Fold {fold_idx}.")
    return train_files, val_files