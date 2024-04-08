# KITS: Inductive Spatio-Temporal Kriging with Increment Training Strategy

### Dependencies

- Python 3.8
- PyTorch 1.8.1
- PyTorch Lightning 1.4.0
- cuda 11.1
```
> conda env create -f env_{ubuntu,windows}.yaml
```

### Datasets

We utilize 8 datasets from different field in this paper:
- Traffic speed datasets:
    - METR-LA
    - PEMS-BAY
    - SEA-LOOP
- Traffic flow dataset:
    - PEMS07
- Air quality datasets (PM2.5):
    - AQI36
    - AQI
- Solar power datasets:
    - NREL-AL
    - NREL-MD

These datasets could be downloaded from this [datasets.zip](https://drive.google.com/file/d/1VQrSLNAr3qr2LAsEK1-_CbBbu6vr0G63/view?usp=sharing), and compressed to the current path.

### Usage

- Run the following commands for training and testing.

- **Training**:
    - E.g., train KITS with missing ratio of 0.5:
      ```
      python train.py --config config/kits/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}.yaml --miss-rate 0.5 --lr 0.0002 --patience 50
      ```

- **Testing**
    - The pretrained KITS (with random seed 1, with missing ratio 0.5) could be downloaded from [final_model.zip](https://drive.google.com/file/d/1uj74MTy6zukWrnwQ_3zoMP67hurtnerP/view?usp=sharing).
    - E.g., test KITS with missing ratio of 0.5:
      ```
      python train.py --config config/kits/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}.yaml --pretrained-model final_model/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}/final/seed1_rm0.5/best.ckpt
      ```

### References

This repo is mainly built based on [grin](https://github.com/Graph-Machine-Learning-Group/grin). Thanks for their great work!
