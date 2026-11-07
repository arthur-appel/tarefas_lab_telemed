### Description ###
A team of Researchers from Quatar University, Doha and Dhaka University along with their connections from Pakistan, Malaysia, in collaboration with doctors, have created the primary dataset with 4 classes(COVID-19, Lung-Opacity, Normal and Viral Pneumonia). The primary dataset has been collected from the Kaggle (COVID-19 Radiography Database [1]). We have added some more contents of Pneumonia and COVID-19 with it from various online resources [2, 3, 4, 5]. Our research work has added one more class with Tuberculosis with the 4 class dataset. We have removed the lateral view Chest X rays and added only the frontal view to the dataset.  The final dataset created using [1, 2 ,3 ,4, 5, 6, 7] has formed a 5 class chest X ray dataset with infectious respiratory diseases. 

### Class Distribution ###
This dataset is a combination of COVID-19, Lung-Opacity, Normal, Viral Pneumonia and Tuberculosis classes.
---------------------------------------
Class Name: Number of samples : Labels
---------------------------------------
COVID-19 :  4,189 : 0
Lung-Opacity : 6,012 : 1
Normal : 10,192 : 2
Viral Pneumonia : 7,397 : 3
Tuberculosis : 4,897 : 4
------------------------------
Total : 32,687
------------------------------

### Splits ###
Splitted into Train, Test and Validation.

### Files Description ###
Each of the .npz files contains a Python dictionary with three keys namely "image", "image_name", and 
"image_label" respectively. The value corresponding to the key "image" is a 4D Numpy array of all image pixel intensity values.
As the name suggests, The value corresponding to the key "image_name" is a numpy array of strings (actual image name) and 
the value corresponding to the key "image_label" is a numpy array of integers (0, 1, 2, 3, or 4).

### Citation ###
Please cite this following articles is you are using this dataset:
TBD

### Reference ###

[1]https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
[2]https://github.com/agchung/Actualmed-COVID-chestxray-datase
[3]https://github.com/agchung/Figure1-COVID-chestxray-dataset
[4]https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
[5]https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
[6]https://www.kaggle.com/raddar/tuberculosis-chest-xrays-shenzhen
[7]https://openi.nlm.nih.gov/faq#collection

### License ###

The dataset follows from the licenses of the original components.