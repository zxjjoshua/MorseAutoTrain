# MorseAutoTrain

### data set
We have 2 data sample in EventData/. you can set set data to run by setting file path in __main__() of data_read.py.

### how to run
In the following example, training_data.out is used as the training data, north_korea_apt_attack_data_debug.out is used as the test data



For training:
```
# In the MorseAutoTrain directory, execute
. experiment_scripts/example_train.sh
```
For testing:
```
# In the MorseAutoTrain directory, execute
. experiment_scripts/example_test.sh
```

### main functions
##### data_read
this function is used for raw data reading, raw data will be tranformed to record object.

##### filenode.py and processnode.py
class files for node

##### event module
there are event_parser.py and event_processor.py.
event_parser.py is for event data parsing, this function maps events to corresponding processing functions in event_processor.py.
event_processor.py contains function for event processing.
