# Finalised_Algorithm_ResNet

Finalised Algorithm with ResNet as training model. Since the aggregation scheme is manual, the structure of aggregating ResNet and aggregating CNN is different.

### Main File Compositions and Functions

##### main.py

Main function to be run in terminal

##### bd_strategy.py

Our algorithm, how clustering and aggregation works

##### mixedbackdoor.py

Extract the dataset and poison it

##### dataset.py

First calls the function in **dataset_preparation.py** to partition the data into training, validation and test datasets. Then calls the functions in **mixbackdoor.py** to poison the data accordingly

##### dataset_preparation.py

Parition dataset according to IID or Non-IID format
