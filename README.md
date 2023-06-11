# Reconstruction of stimulus sentences using deep sentence generation model

This is a repository of codes for Reconstruction of stimulus sentences using deep sentence generation model.

## Dataset

Pereira 2018 Dataset: https://evlab.mit.edu/sites/default/files/documents/index2.html

```
├ vectors_243sentences_dereferencedpronouns.GV42B300.average.txt
├ vectors_384sentences_dereferencedpronouns.GV42B300.average.txt
├─ P01
│   └─ P01
│        ├─ examples_243sentences
│        └─ examples_384sentences
├─ M01
│   └─ M01
│        ├─ examples_243sentences
│        └─ examples_384sentences
：
```

---
## Step 1. Preprocessing
run `python scripts/step1_preprocessing/make_dataset_pereira2018.py [exp_id] [sub_id]`

- Set the path in `config.ini`.  
    ```
    [Pereira2018]
    main_directory = /path/to/directory/
    ```

---
## Step 2. Training (Embedding / Mapping / Reconstruction)
### Model: Optimus
run `python scripts/step2_training/run_generate.py [exp_id] [sub_id] optimus`  
run `python scripts/step2_training/run_predict.py [exp_id] [sub_id] optimus [--reshape] [--feature_selection] [--regressor] [--permute]`

pretrained model's checkpoint: https://textae.blob.core.windows.net/optimus/output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-31250.zip

- Set the path in `config.ini`.  
    ```
    [VAE]
    output_encoder_dir = /path/to/checkpoint-31250/checkpoint-encoder-31250/
    output_decoder_dir = /path/to/checkpoint-31250/checkpoint-decoder-31250/
    checkpoint_dir = /path/to/checkpoint-31250/
    ```

### Model: GloVe
run `python scripts/step2_training/run_predict.py [exp_id] [sub_id] glove [--reshape] [--feature_selection] [--regressor] [--permute]`

- We ued GloVe embedding in Pereira 2018 Dataset.

### Model: BERT
run `python scripts/step2_training/run_generate.py [exp_id] [sub_id] bert`  
run `python scripts/step2_training/run_predict.py [exp_id] [sub_id] bert [--reshape] [--feature_selection] [--regressor] [--permute]`

- We ued pretrained BERT.

---
## Step 3. Evaluation
run `python scripts/step3_evaluation/evaluate.py [model]`  
run `python scripts/step3_evaluation/evaluate_topic.py [model] [--condition]`  
run `python scripts/step3_evaluation/evaluate_bertscore.py [model] [--permute]`  
run `python scripts/step3_evaluation/visualize_allmodel.py`  

---
## Step 4. Post-hoc Analysis
- permutation test
- compute statistics
- analysis for ROI and reconstruction

---
## Batch script
run `python batch/run_batch.py --run`
