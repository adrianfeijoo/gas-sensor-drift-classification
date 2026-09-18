# Data

## Gas Sensor Array Drift Dataset

This project uses the **Gas Sensor Array Drift Dataset** from the
UCI Machine Learning Repository.

The dataset contains 13,910 measurements collected from 16 chemical sensors
exposed to six pure gases at different concentration levels. Measurements
were collected between January 2007 and February 2011 and are divided into
10 acquisition batches to support the study of sensor and concept drift.

Each observation contains 128 continuous features extracted from the sensor
responses and a gas class label.

The original files are stored unchanged under:

`data/raw/gas_sensor_array_drift/`

### Source

Vergara, A. (2012). Gas Sensor Array Drift Dataset.
UCI Machine Learning Repository.

DOI: 10.24432/C5RP6W

License: CC BY 4.0

## Data handling

Files under `raw/` are kept unchanged from the original source.
Any parsing, preprocessing, feature scaling, or feature engineering used in
the experiments is performed programmatically.