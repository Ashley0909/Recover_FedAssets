# FedAssets: Official Code for "Transforming Threats to Assets: Utilizing Backdoor Attack Models in Federated Learning"

## Main Steps 

### 1) Clone Repository

```python
git clone https://github.com/Ashley0909/FedAssets.git
```

### 2) Override Configuration in `conf/base.yaml` if needed

Possible Combinations are:

| Dataset    | num_classes |  target_label | batch_size | num_clients | num_clients_per_round_fit |
| ---------- | :---------: | ------------: | ---------: | ----------: | ------------------------: |
| 'mnist'    |     10     |             9 |         20 |          50 |                        15 |
| 'cifar10'  |     10     |             9 |         20 |          50 |                        15 |
| 'cifar100' |     100     |             9 |         20 |          50 |                        15 |
| 'celeba'   |      3      | 2 ('Smiling") |         20 |          50 |                        15 |


### 3) Run File

```python
python3 main.py
```

## Repository Structure

The following is the structure and key files to make FedAssets work. 

```
├── conf/
│   └── base.yaml               # Configuration file 
├── data/
├── triggers/
│   ├── trigger_10.py
│   ├── trigger_white.py
├── bd_strategy.py              # The file that includes the communication and FedAssets
├── client.py
├── dataset_preparation.py      # Partition the data to good and bad datasets according to IID or Non-IID format
├── dataset.py                  # Imports the data and calls functions in dataset_preparation.py
├── main.py                     # Entrypoint file
├── model.py
├── server.py
│ 
```
