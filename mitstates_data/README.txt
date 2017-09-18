This is the splits data used for the MIT States dataset in our paper (specifically Table 1 of our paper).
It has 700 unseen pairs of attributes and objects for the testing set
The following files are included
- mit_image_data.pklz: contains the image_ids and filenames used in the label
  files
- split_meta_data.pklz: contains the splits used for training and testing
- split_labels_train.pklz: training labels
- split_labels_test.pklz: testing labels
- im_utils.py: python code to read the pklz files. `import im_utils; data_dict  = im_utils.load(filename)`
