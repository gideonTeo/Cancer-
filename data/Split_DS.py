# install 'pip install split-folders' for small files
# install 'pip install split-folders tqdm' for larger files (tqdm for visual updates for copying files)

# Author: Zhi Kai Teo
# Date Created: 19/07/2021
import os
import splitfolders  # or import split_folders

input_folder = os.getcwd() + "/data/archive"
#input_folder = os.getcwd() + "/data/Dataset2525/train"
print(input_folder)
output_folder = os.getcwd() + "/data/Dataset120507"
# output_folder = os.getcwd() + "\\data\\Dataset0502_test"    # splitting smaller dataset for unit testing
# output_folder = os.getcwd() + "\\data\\Dataset5050"
# output_folder = os.getcwd() + "\\data\\Dataset2525"

# Split with a ratio - ratio to 70% (training), 20% (validation), 10% (test)
"""
50% used for training the data and update the weights.
20% validation data is used as part of the training but only to validate and reports a metrics after each epoch.
30% testing to test the completed trained model.
"""
splitfolders.ratio(input_folder, output=output_folder, seed=50, ratio=(.5, .2, .3), group_prefix=None) # default values
# splitfolders.ratio(input_folder, output=output_folder, seed=50, ratio=(.05, .02, .93), group_prefix=None) # default values
# splitfolders.ratio(input_folder, output=output_folder, seed=50, ratio=(.5, .5), group_prefix=None) # default values

########################################################
#                      Using this                      #
########################################################
# Split val/test with a fixed number of items e.g. 1000 and 500 for each set.
# To only split into training and validation set, use a single number to `fixed`
# FYI: training folder must be adjusted manually from the folder. This code only adjust Validation and Test dataset.
# splitfolders.fixed(input_folder, output="Dataset", seed=50, fixed=(1000, 500), oversample=False, group_prefix=None) # default values

