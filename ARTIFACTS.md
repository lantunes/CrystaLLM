Artifacts
=========

All trained models, training sets, and artifacts generated by the models have been uploaded to Zenodo. The files are 
publicly accessible at: [https://zenodo.org/records/10642388](https://zenodo.org/records/10642388). All files are 
released under the [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

Each file listed below can be downloaded using the `download.py` script. For example, to download `cifs_v1_val.pkl.gz`:
```shell
python bin/download.py cifs_v1_val.pkl.gz
```

### Main Dataset

| Name                       | Description                                                                                  |  Download Link                                                                            |
|----------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| cifs_v1_orig.tar.gz        | The original CIF file dataset containing 3,551,492 symmetrized CIF files.                    | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_orig.tar.gz)        |
| cifs_v1_orig.pkl.gz        | The contents of `cifs_v1_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).    | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_orig.pkl.gz)        |
| cifs_v1_dedup.tar.gz       | The deduplicated original CIF dataset, containing 2,285,914 symmetrized CIF files.           | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_dedup.tar.gz)       |
| cifs_v1_dedup.pkl.gz       | The contents of `cifs_v1_dedup.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).   | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_dedup.pkl.gz)       |
| cifs_v1_prep.tar.gz        | The deduplicated and pre-processed original CIF dataset, containing 2,285,719 CIF files.     | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_prep.tar.gz)        |
| cifs_v1_prep.pkl.gz        | The contents of `cifs_v1_prep.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).    | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_prep.pkl.gz)        |
| cifs_v1_train.tar.gz       | The training split of the main dataset, containing 2,047,889 CIF files.                      | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_train.tar.gz)       |
| cifs_v1_train.pkl.gz       | The contents of `cifs_v1_train.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).   | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_train.pkl.gz)       |
| cifs_v1_val.tar.gz         | The validation split of the main dataset, containing 227,544 CIF files.                      | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_val.tar.gz)         |
| cifs_v1_val.pkl.gz         | The contents of `cifs_v1_val.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).     | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_val.pkl.gz)         |
| cifs_v1_test.tar.gz        | The test split of the main dataset, containing 10,286 CIF files.                             | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_test.tar.gz)        |
| cifs_v1_test.pkl.gz        | The contents of `cifs_v1_test.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).    | [download &#x2193;](https://zenodo.org/records/10642388/files/cifs_v1_test.pkl.gz)        |
| tokens_v1_all.tar.gz       | The tokens of the complete main dataset.                                                     | [download &#x2193;](https://zenodo.org/records/10642388/files/tokens_v1_all.tar.gz)       |
| tokens_v1_train_val.tar.gz | The tokens of the training and validation sets of the main dataset.                          | [download &#x2193;](https://zenodo.org/records/10642388/files/tokens_v1_train_val.tar.gz) |
| starts_v1_train.pkl        | The start indices for the tokenized training set structures of the main dataset.             | [download &#x2193;](https://zenodo.org/records/10642388/files/starts_v1_train.pkl)        |
| starts_v1_val.pkl          | The start indices for the tokenized validation set structures of the main dataset.           | [download &#x2193;](https://zenodo.org/records/10642388/files/starts_v1_val.pkl)          |
| challenge_set_v1.zip       | The structures of the challenge set.                                                         | [download &#x2193;](https://zenodo.org/records/10642388/files/challenge_set_v1.zip)       |

### Pre-trained Models

| Name                                    | Description                                                                                                | Download Link                                                                                          |
| ----------------------------------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| crystallm_v1_small.tar.gz               | Model with small architecture trained on the full main dataset.                                            | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz)               |
| crystallm_v1_large.tar.gz               | Model with large architecture trained on the full main dataset.                                            | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_v1_large.tar.gz)               |
| crystallm_perov_5_small.tar.gz          | Model with small architecture trained on the Perov-5 training set only.                                    | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_perov_5_small.tar.gz)          |
| crystallm_perov_5_large.tar.gz          | Model with large architecture trained on the Perov-5 training set only.                                    | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_perov_5_large.tar.gz)          |
| crystallm_carbon_24_small.tar.gz        | Model with small architecture trained on the Carbon-24 training set only.                                  | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_carbon_24_small.tar.gz)        |
| crystallm_carbon_24_large.tar.gz        | Model with large architecture trained on the Carbon-24 training set only.                                  | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_carbon_24_large.tar.gz)        |
| crystallm_mp_20_small.tar.gz            | Model with small architecture trained on the MP-20 training set only.                                      | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_mp_20_small.tar.gz)            |
| crystallm_mp_20_large.tar.gz            | Model with large architecture trained on the MP-20 training set only.                                      | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_mp_20_large.tar.gz)            |
| crystallm_mpts_52_small.tar.gz          | Model with small architecture trained on the MPTS-52 training set only.                                    | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_mpts_52_small.tar.gz)          |
| crystallm_mpts_52_large.tar.gz          | Model with large architecture trained on the MPTS-52 training set only.                                    | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_mpts_52_large.tar.gz)          |
| crystallm_v1_minus_mpts_52_small.tar.gz | Model with small architecture trained on the full main dataset minus the MPTS-52 test and validation sets. | [download &#x2193;](https://zenodo.org/records/10642388/files/crystallm_v1_minus_mpts_52_small.tar.gz) |

### Perov-5 Dataset

| Name                        | Description                                                                                     | Download Link                                                                              |
|-----------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| perov_5_train_orig.tar.gz   | The original CIF files of the Perov-5 training set (symmetrized).                               | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_train_orig.tar.gz)   |
| perov_5_train_orig.pkl.gz   | The contents of `perov_5_train_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF). | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_train_orig.pkl.gz)   |
| perov_5_train_prep.pkl.gz   | The pre-processed CIF files of the Perov-5 training set.                                        | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_train_prep.pkl.gz)   |
| perov_5_val_orig.tar.gz     | The original CIF files of the Perov-5 validation set (symmetrized).                             | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_val_orig.tar.gz)     |
| perov_5_val_orig.pkl.gz     | The contents of `perov_5_val_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).   | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_val_orig.pkl.gz)     |
| perov_5_val_prep.pkl.gz     | The pre-processed CIF files of the Perov-5 validation set.                                      | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_val_prep.pkl.gz)     |
| perov_5_test_orig.tar.gz    | The original CIF files of the Perov-5 test set (symmetrized).                                   | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_test_orig.tar.gz)    |
| perov_5_test_orig.pkl.gz    | The contents of `perov_5_test_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).  | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_test_orig.pkl.gz)    |
| perov_5_test_prep.pkl.gz    | The pre-processed CIF files of the Perov-5 test set.                                            | [download &#x2193;](https://zenodo.org/records/10642388/files/perov_5_test_prep.pkl.gz)    |
| tokens_perov_5.tar.gz       | The tokens of the Perov-5 training and validation sets.                                         | [download &#x2193;](https://zenodo.org/records/10642388/files/tokens_perov_5.tar.gz)       |
| starts_perov_5_train.pkl    | The start indices for the tokenized training set structures of the Perov-5 training set.        | [download &#x2193;](https://zenodo.org/records/10642388/files/starts_perov_5_train.pkl)    |
| prompts_perov_5_test.tar.gz | Text files containing prompts derived from the Perov-5 test set.                                | [download &#x2193;](https://zenodo.org/records/10642388/files/prompts_perov_5_test.tar.gz) |

### Carbon-24 Dataset

| Name                          | Description                                                                                       | Download Link                                                                                |
|-------------------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| carbon_24_train_orig.tar.gz   | The original CIF files of the Carbon-24 training set (symmetrized).                               | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_train_orig.tar.gz)   |
| carbon_24_train_orig.pkl.gz   | The contents of `carbon_24_train_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF). | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_train_orig.pkl.gz)   |
| carbon_24_train_prep.pkl.gz   | The pre-processed CIF files of the Carbon-24 training set.                                        | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_train_prep.pkl.gz)   |
| carbon_24_val_orig.tar.gz     | The original CIF files of the Carbon-24 validation set (symmetrized).                             | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_val_orig.tar.gz)     |
| carbon_24_val_orig.pkl.gz     | The contents of `carbon_24_val_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).   | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_val_orig.pkl.gz)     |
| carbon_24_val_prep.pkl.gz     | The pre-processed CIF files of the Carbon-24 validation set.                                      | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_val_prep.pkl.gz)     |
| carbon_24_test_orig.tar.gz    | The original CIF files of the Carbon-24 test set (symmetrized).                                   | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_test_orig.tar.gz)    |
| carbon_24_test_orig.pkl.gz    | The contents of `carbon_24_test_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).  | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_test_orig.pkl.gz)    |
| carbon_24_test_prep.pkl.gz    | The pre-processed CIF files of the Carbon-24 test set.                                            | [download &#x2193;](https://zenodo.org/records/10642388/files/carbon_24_test_prep.pkl.gz)    |
| tokens_carbon_24.tar.gz       | The tokens of the Carbon-24 training and validation sets.                                         | [download &#x2193;](https://zenodo.org/records/10642388/files/tokens_carbon_24.tar.gz)       |
| starts_carbon_24_train.pkl    | The start indices for the tokenized training set structures of the Carbon-24 training set.        | [download &#x2193;](https://zenodo.org/records/10642388/files/starts_carbon_24_train.pkl)    |
| prompts_carbon_24_test.tar.gz | Text files containing prompts derived from the Carbon-24 test set.                                | [download &#x2193;](https://zenodo.org/records/10642388/files/prompts_carbon_24_test.tar.gz) |

### MP-20 Dataset

| Name                      | Description                                                                                   | Download Link                                                                            |
|---------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| mp_20_train_orig.tar.gz   | The original CIF files of the MP-20 training set (symmetrized).                               | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_train_orig.tar.gz)   |
| mp_20_train_orig.pkl.gz   | The contents of `mp_20_train_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF). | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_train_orig.pkl.gz)   |
| mp_20_train_prep.pkl.gz   | The pre-processed CIF files of the MP-20 training set.                                        | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_train_prep.pkl.gz)   |
| mp_20_val_orig.tar.gz     | The original CIF files of the MP-20 validation set (symmetrized).                             | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_val_orig.tar.gz)     |
| mp_20_val_orig.pkl.gz     | The contents of `mp_20_val_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).   | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_val_orig.pkl.gz)     |
| mp_20_val_prep.pkl.gz     | The pre-processed CIF files of the MP-20 validation set.                                      | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_val_prep.pkl.gz)     |
| mp_20_test_orig.tar.gz    | The original CIF files of the MP-20 test set (symmetrized).                                   | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_test_orig.tar.gz)    |
| mp_20_test_orig.pkl.gz    | The contents of `mp_20_test_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).  | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_test_orig.pkl.gz)    |
| mp_20_test_prep.pkl.gz    | The pre-processed CIF files of the MP-20 test set.                                            | [download &#x2193;](https://zenodo.org/records/10642388/files/mp_20_test_prep.pkl.gz)    |
| tokens_mp_20.tar.gz       | The tokens of the MP-20 training and validation sets.                                         | [download &#x2193;](https://zenodo.org/records/10642388/files/tokens_mp_20.tar.gz)       |
| starts_mp_20_train.pkl    | The start indices for the tokenized training set structures of the MP-20 training set.        | [download &#x2193;](https://zenodo.org/records/10642388/files/starts_mp_20_train.pkl)    |
| prompts_mp_20_test.tar.gz | Text files containing prompts derived from the MP-20 test set.                                | [download &#x2193;](https://zenodo.org/records/10642388/files/prompts_mp_20_test.tar.gz) |

### MPTS-52 Dataset

| Name                           | Description                                                                                     | Download Link                                                                                 |
|--------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| mpts_52_train_orig.tar.gz      | The original CIF files of the MPTS-52 training set (symmetrized).                               | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_train_orig.tar.gz)      |
| mpts_52_train_orig.pkl.gz      | The contents of `mpts_52_train_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF). | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_train_orig.pkl.gz)      |
| mpts_52_train_prep.pkl.gz      | The pre-processed CIF files of the MPTS-52 training set.                                        | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_train_prep.pkl.gz)      |
| mpts_52_val_orig.tar.gz        | The original CIF files of the MPTS-52 validation set (symmetrized).                             | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_val_orig.tar.gz)        |
| mpts_52_val_orig.pkl.gz        | The contents of `mpts_52_val_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).   | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_val_orig.pkl.gz)        |
| mpts_52_val_prep.pkl.gz        | The pre-processed CIF files of the MPTS-52 validation set.                                      | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_val_prep.pkl.gz)        |
| mpts_52_test_orig.tar.gz       | The original CIF files of the MPTS-52 test set (symmetrized).                                   | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_test_orig.tar.gz)       |
| mpts_52_test_orig.pkl.gz       | The contents of `mpts_52_test_orig.tar.gz` as a serialized Python list of 2-tuples: (ID, CIF).  | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_test_orig.pkl.gz)       |
| mpts_52_test_prep.pkl.gz       | The pre-processed CIF files of the MPTS-52 test set.                                            | [download &#x2193;](https://zenodo.org/records/10642388/files/mpts_52_test_prep.pkl.gz)       |
| tokens_mpts_52.tar.gz          | The tokens of the MPTS-52 training and validation sets.                                         | [download &#x2193;](https://zenodo.org/records/10642388/files/tokens_mpts_52.tar.gz)          |
| tokens_v1_minus_mpts_52.tar.gz | The tokens of the full main dataset minus the MPTS-52 validation and test sets.                 | [download &#x2193;](https://zenodo.org/records/10642388/files/tokens_v1_minus_mpts_52.tar.gz) |
| starts_mpts_52_train.pkl       | The start indices for the tokenized training set structures of the MPTS-52 training set.        | [download &#x2193;](https://zenodo.org/records/10642388/files/starts_mpts_52_train.pkl)       |
| prompts_mpts_52_test.tar.gz    | Text files containing prompts derived from the MPTS-52 test set.                                | [download &#x2193;](https://zenodo.org/records/10642388/files/prompts_mpts_52_test.tar.gz)    |

### Generated Benchmark CIF Files

| Name                                  | Description                                                                                                                                                                        | Download Link                                                                                        |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| gen_perov_5_small_raw.tar.gz          | CIF files generated with the Perov-5 small model starting from the Perov-5 test set prompts (_n=20_).                                                                              | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_perov_5_small_raw.tar.gz)          |
| gen_perov_5_small.tar.gz              | Pre-processed CIF files generated with the Perov-5 small model starting from the Perov-5 test set prompts (_n=20_).                                                                | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_perov_5_small.tar.gz)              |
| gen_perov_5_large_raw.tar.gz          | CIF files generated with the Perov-5 large model starting from the Perov-5 test set prompts (_n=20_).                                                                              | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_perov_5_large_raw.tar.gz)          |
| gen_perov_5_large.tar.gz              | Pre-processed CIF files generated with the Perov-5 large model starting from the Perov-5 test set prompts (_n=20_).                                                                | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_perov_5_large.tar.gz)              |
| gen_carbon_24_small_raw.tar.gz        | CIF files generated with the Carbon-24 small model starting from the Carbon-24 test set prompts (_n=20_).                                                                          | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_carbon_24_small_raw.tar.gz)        |
| gen_carbon_24_small.tar.gz            | Pre-processed CIF files generated with the Carbon-24 small model starting from the Carbon-24 test set prompts (_n=20_).                                                            | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_carbon_24_small.tar.gz)            |
| gen_carbon_24_large_raw.tar.gz        | CIF files generated with the Carbon-24 large model starting from the Carbon-24 test set prompts (_n=20_).                                                                          | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_carbon_24_large_raw.tar.gz)        |
| gen_carbon_24_large.tar.gz            | Pre-processed CIF files generated with the Carbon-24 large model starting from the Carbon-24 test set prompts (_n=20_).                                                            | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_carbon_24_large.tar.gz)            |
| gen_mp_20_small_raw.tar.gz            | CIF files generated with the MP-20 small model starting from the MP-20 test set prompts (_n=20_).                                                                                  | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_mp_20_small_raw.tar.gz)            |
| gen_mp_20_small.tar.gz                | Pre-processed CIF files generated with the MP-20 small model starting from the MP-20 test set prompts (_n=20_).                                                                    | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_mp_20_small.tar.gz)                |
| gen_mp_20_large_raw.tar.gz            | CIF files generated with the MP-20 large model starting from the MP-20 test set prompts (_n=20_).                                                                                  | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_mp_20_large_raw.tar.gz)            |
| gen_mp_20_large.tar.gz                | Pre-processed CIF files generated with the MP-20 large model starting from the MP-20 test set prompts (_n=20_).                                                                    | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_mp_20_large.tar.gz)                |
| gen_mpts_52_large_raw.tar.gz          | CIF files generated with the MPTS-52 large model starting from the MPTS-52 test set prompts (_n=20_).                                                                              | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_mpts_52_large_raw.tar.gz)          |
| gen_mpts_52_large.tar.gz              | Pre-processed CIF files generated with the MPTS-52 large model starting from the MPTS-52 test set prompts (_n=20_).                                                                | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_mpts_52_large.tar.gz)              |
| gen_v1_minus_mpts_52_small_raw.tar.gz | CIF files generated with the small model trained on the full dataset minus the MPTS-52 test and validation sets, starting from the MPTS-52 test set prompts (_n=20_).              | [download &#x2193;](https://zenodo.org/records/10642388/files/gen_v1_minus_mpts_52_small_raw.tar.gz) |
| gen_v1_minus_mpts_52_small.tar.gz     | Pre-processed CIF files generated with the small model trained on the full dataset minus the MPTS-52 test and validation sets, starting from the MPTS-52 test set prompts (_n=20_).| [download &#x2193;](https://zenodo.org/records/10642388/files/gen_v1_minus_mpts_52_small.tar.gz)     |