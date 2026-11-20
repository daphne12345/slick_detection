# Detecting Sea Surface Slicks using Automated Machine Learning

Official code for the paper ["Detecting Sea Surface Slicks using Automated Machine Learning"](https://ieeexplore.ieee.org/document/10244633). We classify whether a sea surface slick is visible on an optical image, comparing auto-sklearn on hand-crafted image features with AutoKeras on the raw images.

## Code

All scripts are in `src/` and are run from the repository root with `python -m src.<script>`.

| Paper | Scripts |
|-------|---------|
| III Datasets | `download_data`, `image_quality`, `wind_speed_filter`, `labels`, `merge_datasets` |
| IV-B, V-A auto-sklearn | `features`, `feature_ttest`, `make_smote_dataset`, `run_autosklearn` |
| V-B to V-D AutoKeras | `prepare_datasets`, `run_autokeras`, `train_keras_model` |
| Figures | `plots` |
| Inference | `find_slicks` |

## Installation

```bash
conda create -n slicks python=3.8 && conda activate slicks
pip install -r requirements.txt
```

auto-sklearn only runs on Linux and conflicts with TensorFlow, so install it with
`pip install auto-sklearn` in a separate environment.

## Usage

The image datasets are not included. Image lists (`python -m src.labels list-files`) are text files
with paths separated by `", "`. Label files (`python -m src.labels list-labels`) map image names to
`True`/`False`.

```bash
# auto-sklearn
python -m src.features --im_dir images/train.csv --saveto images/slicks_features.csv
python -m src.feature_ttest --im_dir images/train.csv
python -m src.make_smote_dataset --filter_method f
python -m src.run_autosklearn --filter_method f --metric f1 --ensemble_size 50

# AutoKeras (zip files are created with src.prepare_datasets.Dataset.create_dataset)
python -m src.run_autokeras --zip_file_train train.zip --zip_file_test test.zip \
    --load_labels_from labels.csv --saveto results --metric f1 --train_new True --project_name ak
python -m src.train_keras_model --zip_file_train train.zip --zip_file_test test.zip \
    --load_labels_from labels.csv --saveto results --model_dir results/autokeras.h5 \
    --tune_hyperparameters True

# classify new images
python -m src.find_slicks -d path/to/images --load_from path/to/final_classifier.h5
```

## Citation

```bibtex
@inproceedings{ossenbeck2023slicks,
  title     = {Detecting Sea Surface Slicks using Automated Machine Learning},
  author    = {Ossenbeck, Milena and Theodorakopoulos, Daphne and Schneemann, J{\"o}rge and
               Ferdinand, Oliver and Ribas-Ribas, Mariana},
  booktitle = {OCEANS},
  year      = {2023}
}
```
