CIND-860 Capstone Project: Detecting Breast Cancer from Mammogram Images Using a CNN

Robert M. Pineau
941-049-371

Supervisor:  Dr. Ceni Babaoglu

The Data used for this project was from the Kaggle/RSNA "Screening Mammography Breast Cancer Detection" competition (Nov 2022 - Apr 2023)
https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data


**********************************************************************************************************************************


Files in this project:

CIND860_data_desc.ipynb

CIND860_image_prep.ipynb

CIND860_shuffle_split_copy.ipynb

CIND860_augment_train.ipynb

CIND860_build_numpy_batches.ipynb

CIND860_CNN.ipynb
**********************************************************************************************************************************


Due to the logistics of using a dataset containing images totalling 271 GB(over even the GOOGLE One/Google Colab Pro limit)
The python utlilites:

CIND860_image_prep.ipynb

CIND860_shuffle_split_copy.ipynb

CIND860_augment_train.ipynb

CIND860_build_numpy_batches.ipynb

Were all run on a home PC utilizing a 4TB drive.   While the source code is in .ipynb format,
no output is present due to they were run with python on the commandline.

These utilities were created custom for this project, and run serially,
The first one acting on the dicom formatted high resolution mammogram images, until the last one was run,
providing the files train.tgz, validate.tgz, test.tgz, which were uploaded to Goole Drive, for CNN creation, training, validation, and testing.
**********************************************************************************************************************************



Stages(Section) of the project(software):

Initial descriptive statistics portion(CIND860_data_desc.ipynb)

Image Prep(CIND860_image_prep.ipynb)

Train/Validate/Split Data(CIND860_shuffle_split_copy.ipynb)

Augment Training Data(CIND860_augment_train.ipynb)

Build numpy based batches of data to be used for training/validation and testing(CIND860_build_numpy_batches.ipynb)

CNN Model Creation/Training/Validation(CIND860_CNN.ipynb)
**********************************************************************************************************************************




