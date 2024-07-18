# HELLO!
This is an official implement of Affective EEG-Based Person Identification Across Electrode Configurations.



## Dataset

This folder contains the .py file explaining how to convert the SEED/FACED datasets into data suitable for person identification (PI) tasks.

You can see that the FACED dataset can be split into 30 channels and 62 channels, and the same applies to the SEED dataset.

If you want to use EQ-VA, you should run the script to generate FACED 30 channels and SEED 62 channels.

If you want to compare different methods, you should use data with the same number of channels.



## shape[bs,n_channels,n_second,de_features]

This folder includes the code about training and validation of model.