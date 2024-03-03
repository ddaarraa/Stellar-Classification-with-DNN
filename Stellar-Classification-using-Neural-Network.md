# 1. About Dataset :
Context : 

              In astronomy, stellar classification is the classification of stars based on their spectral characteristics. 
       The classification scheme of galaxies, quasars, and stars is one of the most fundamental in astronomy. 
       The early cataloguing of stars and their distribution in the sky has led to the understanding that they make up our own galaxy and, 
       following the distinction that Andromeda was a separate galaxy to our own, numerous galaxies began to be surveyed as more powerful 
       telescopes were built. This datasat aims to classificate stars, galaxies, and quasars based on their spectral characteristics.

*Content*

       The data consists of 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey). Every observation
       is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar.

- obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS
- alpha = Right Ascension angle (at J2000 epoch)
- delta = Declination angle (at J2000 epoch)
- u = Ultraviolet filter in the photometric system
- g = Green filter in the photometric system
- r = Red filter in the photometric system
- i = Near Infrared filter in the photometric system
- z = Infrared filter in the photometric system
- run_ID = Run Number used to identify the specific scan
- rereun_ID = Rerun Number to specify how the image was processed
- cam_col = Camera column to identify the scanline within the run
- field_ID = Field number to identify each field
- spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
- class = object class (galaxy, star or quasar object)
- redshift = redshift value based on the increase in wavelength
- plate = plate ID, identifies each plate in SDSS
- MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
- fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation

#
# 2.  Data Visualization


<!-- ![Unknown](https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/114089025/80ebdfc5-40e7-4eba-8c89-757af51c0904) -->

we're displaying the sample amount of every class ,
since our dataset is not yet in a balance condition, we're going to go through some process to make them balance.




![Unknown](https://github.com/640710505/Sellar-Classification-using-Neural-Network/assets/114089025/dfadc634-c486-4f6f-9fa8-3cad6e3d14b0)
#
Here we go, code of data balancing. In this case , we chose to randomly drop out some value of the majority class- Galaxy class - to let its values decrease to an amount equaled to the one in QSO class

<details>
<summary><strong>Code</strong></summary>

```python
import pandas as pd
import numpy as np

# Check class distribution
class_distribution = dataset['class'].value_counts()
print("Class Distribution Before Balancing:")
print(class_distribution)

# Determine the majority class
majority_class = class_distribution.idxmax()

# Get the number of samples in the majority class
majority_class_count = class_distribution[majority_class]

# Find indices of majority class samples
majority_indices = dataset[dataset['class'] == majority_class].index

# Randomly select majority class samples to drop
drop_indices = np.random.choice(majority_indices, size=majority_class_count - minority_class_count, replace=False)

# Drop selected majority class samples
balanced_df = dataset.drop(drop_indices)

# Check class distribution after balancing
balanced_class_distribution = balanced_df['class'].value_counts()
print("\nClass Distribution After Balancing:")
print(balanced_class_distribution)
```
</details>
<details>

<summary><strong>Output</strong></summary>

```
Class Distribution Before Balancing:
GALAXY    59445
STAR      21594
QSO       18961
Name: class, dtype: int64

Class Distribution After Balancing:
STAR      21594
GALAXY    18961
QSO       18961
Name: class, dtype: int64
```
</details>




#
And Now we're going to display how the distribution of samples of every class would look like

<details>
<summary><strong>Code</strong></summary>

```python
import matplotlib.pyplot as plt
from collections import Counter

# Counting the frequency of each class in the dataset
element_count = Counter(balanced_df['class'])

# Extracting elements and their counts
elements = list(element_count.keys())
counts = list(element_count.values())

# Plotting
plt.figure(figsize=(8, 6))
bars = plt.bar(elements, counts, color='blue')
plt.xlabel('Elements', color="white")  # Change the color of the x-axis label
plt.ylabel('Frequency', color="white")  # Change the color of the y-axis label
plt.title('Distinctive Elements Frequency', color="white")  # Change the color of the title

# Change the font color of the tick labels on the x and y axes
plt.xticks(color="white")
plt.yticks(color="white")

# Change the font color of the counts on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom', color="black")

plt.show()
```
</details>

![Unknown](https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/114089025/9c0a3c05-6f9e-453e-a80d-01a776fe7405)
#










### Filtering important features, Using Logistics Regression

- First assign   feature to **X** and target to **Y**  

```python
X = data[['obj_ID','alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID',
       'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift',
       'plate', 'MJD', 'fiber_ID']]
Y = data[['class']]
```
#

- Spliting the dataset into training set and testing set by the ratio of 80 : 20
```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, 
                                                 random_state=42,
                                                 shuffle = True,
                                                 stratify= Y)
```
#

- Perform StandardScaler to scaling the feature into the same scale
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test) 
```
#

- Create and train LogisticRegression model
```python
model = LogisticRegression()
model.fit(X_train_std,Y_train)
```
#

- Check model score
```python
model.score(X_test_std,Y_test)
```

<details>
<summary><strong>Output</strong></summary><br>

```
0.948473816858023
```
</details>

#

- Check coefficients to observe important features
```python
coefficients = model.coef_
print(coefficients)
```
<details>
  <summary><strong>Output</strong></summary><br>

```
[[ 1.45477542e-02 -1.23103249e-02 -4.53120066e-02  7.01904752e+00
   4.68105844e+00  1.04455625e+00 -1.55975245e+00 -9.38051743e-01
   1.45520219e-02  0.00000000e+00 -4.32997959e-02  4.53115258e-02
   3.21639936e-01  1.47975574e+01  3.21640912e-01 -7.00410253e-01
   3.47263277e-02]
 [ 2.28612863e-02  8.05451884e-02  1.89388215e-01 -8.30274338e+00
  -2.64094232e+00 -2.85728341e+00  2.66957060e+00  5.55446057e+00
   2.28643623e-02  0.00000000e+00 -3.14602773e-02  2.28082408e-02
  -9.72926616e-02  1.94403139e+01 -9.72935111e-02 -7.41901037e-02
   1.46102515e-02]
 [-3.74090405e-02 -6.82348635e-02 -1.44076209e-01  1.28369587e+00
  -2.04011611e+00  1.81272715e+00 -1.10981815e+00 -4.61640883e+00
  -3.74163842e-02  0.00000000e+00  7.47600732e-02 -6.81197666e-02
  -2.24347275e-01 -3.42378713e+01 -2.24347401e-01  7.74600357e-01
  -4.93365792e-02]]
```
</details>

#

### Coefficient Graph
![featureimportant](https://github.com/640710505/Sellar-Classification-using-Neural-Network/assets/141728733/10905af0-4587-4fcf-bc27-d759f84fc7d1)

Base on graph, indicated that features like "u" , "g", "r", "i", "z" and "redshift" got more coefficient score than anothers...

From now on, we're going to work through our project with these useful filtered features only. 

```python

balanced_df_important_feature_only = balanced_df[['u', 'g', 'r', 'i', 'z', 'redshift','class']]
```
After that we export dataset that only have feature important to CSV
```python
df = pd.DataFrame(balanced_df_important_feature_only)

# Specify the path where you want to save the CSV file
csv_file_path = 'balanced_df_important_feature_only.csv'

# Export the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)
``` 

#
Displaying the Corelation of Every feature :

![Unknown](https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/114089025/80ebdfc5-40e7-4eba-8c89-757af51c0904) 

#









# 3. Feature Extraction

Following the "feature important" filtering, there're currently only 6 features existing.

We want more!!!!

We're going to generate 3 more features :
       - distance 
       - velocity
       - Magnitude
  
The Feature Extraction Process will achieve by following the steps displaying the graph below :

  
![Dataset 1 0 preparation](https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/114089025/39f4462b-9e52-4f1c-b3b3-9c46c49697cb)

We're going to use the redshift value from the orginal dataset to find Velocity and Distance through the equations from Hubble's law
which is not going to deeply explain here in this project.

Boom! now we're having version 1.1 of dataset...
what's next...

We suppose to Extract another new feautre- Magnitude, But how could we achieve that...

Since the Equations we found online , aren't simply eased to be used...due to the complexity of the equation

We fortunately discovered some datasets of star and Galaxy-Quosa, that brought us the relation between some existed feature in our version 1.1 dataset and the Magnitude value...

We're proudly , going to use those datasets to train two seperated DNN models ( one for star, and another one for Galaxy and Quasa combined) in order to predict the new Magnitude value and form a new and final version of our dataset 

![Stellar Dataset for Classification 1 1](https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/114089025/284c0785-93a4-4429-814f-94637fd874ca)

### 3.1 DNN model to predict Magnitude value for Star

This one, the function to predict Magnitude value of the Stars
```python
  def Magnitude_predict_star(distance,ultraviolet):
        
        df = pd.read_csv("dataset/Star39552_balanced.csv")
        X = df[['Plx','B-V']] #distance between star and earth , color index
        y = df['Amag'] #Absolute magnitude

        sc = StandardScaler()
        sc.fit(X)

        model = load_model('star_ABMg_prediction.h5')
        #this one we use our pre-trained model , that is not gonna be displayed in this project
        
        print('distance :', distance )
        print('utraviolet :', ultraviolet)
                
        X_predict = [[distance,ultraviolet]]
        X_predict = sc.transform(X_predict)
        predict = model.predict(X_predict)
        print("-------------------------------------------------------------")
        
        return predict
```
#

### 3.2 DNN model to predict Magnitude value for Galaxy & Quasa

here, we're presenting the function to predict the Magnitude value of Galaxy and Quasa

```python

class galaxy_model:
    df = pd.read_csv("dataset/COMBO17.csv")
    X = df[['e.W571FS','W420FE','W914FE','Mcz']]
    #green_filter = e.W571FS, ultra_violet = 'W420FE', infra red = 'W914FE', redshift = 'Mcz'
    y = df['UjMAG']

    sc = StandardScaler()
    sc.fit(X)

    model2 = load_model('galaxy_ABMg_prediction.h5')
    # this one is also a pre-trained model 
    
    def predict(self,g_filter , ultra_violet ,infra_red, redshift) :
       
        print("g :", g_filter)
        X_predict2 = [[g_filter , ultra_violet , infra_red, redshift]]
        X_predict = galaxy_model.sc.fit_transform(X_predict2)
    
        predict2 = galaxy_model.model2.predict(X_predict)
        print("-------------------------------------------------------------")
        return predict2
```
#
We predict those Magnitude values and add them to the new column of our dataset 

Then Final Dataset would look like this :

![423599841_1142816160225180_7115163343242874335_n](https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/114089025/1318f17f-202e-4889-bac3-6eef70efeda0)

# 4. Deep Neural Network Model for Final Classification
All set!
Here we come, we've arrived at the final stage of the project which is where we're going to build a final DNN model to classify our outer space luminios objects...

We'll explain through the process step by step :

- Assign feature to **X** and target to **y** with the new and final dataset
```python
X = dataset[['u', 'g', 'r', 'i', 'z', 'redshift','distance', 'NewAMagnitude']]
y = dataset[['class']]
```
#
- Spliting the dataset into training set and testing set by the ratio of 80 : 20
```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42, 
                                                               shuffle=True)
```
#
- Perform one-hot encoder to transform target classes to vector for training a neural network model
```python
y_train_one_hot = to_categorical(y_train, num_classes=3)
# num class = 3 following the final product of prediction
y_test_one_hot = to_categorical(y_test, num_classes=3)
# one hot encoder's product should be in shape of vector with 3 binary number elements
# it should be look like this -> class 0 : [1, 0, 0]
#                                class 1 : [0, 1, 0]
#                                class 2 : [0, 0, 1]
print(y_train_one_hot.shape)
```
<details>
<summary><strong>Output</strong></summary>
<br>

```python

(47612, 3)
# 47612 row, and every row consisting of one list with 3 values
```
</details>

# 
- Build Deep Neural Network Model <br>
=> 1 input layer with 2000 nodes and input_shape = 8 features<br>
=> 3 hidden layer with 1000,100,10 nodes <br>
=> 1 output layer with 3 nodes (we have 3 target classes)

```python
from keras import Sequential
from keras.layers import Dense
from keras.backend import clear_session 
model = Sequential()
model.add(Dense(units=2000, activation='relu', input_shape=(8,)))
model.add(Dense(units=1000 , activation='relu'))
model.add(Dense(units=100 , activation='relu'))
model.add(Dense(units= 10 , activation='relu'))
#using ReLu activation function to produce 0 or 1 output
model.add(Dense(units=3, activation='sigmoid'))
#output of every node here would be lay between 0-1
model.summary()
```
<img width="500" alt="Model sequential" src="https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/141728733/c967ad6f-ae21-44b0-b58e-c90d596f6e87">

#
- Setup and Train model <br>
=> optimizer = 'adam' to make learning rate can be adaptive <br>
=> loss = 'categorical_crossentropy' is for multi classes classif <br>
=> epochs = 9 (9 iteration) <br>
=> batch_size = 60 (used samples in each iteration)
```python
model.compile(optimizer="adam", loss='categorical_crossentropy')
history = model.fit(X_train, y_train_one_hot, epochs=9, batch_size = 60, validation_data=(X_test, y_test_one_hot))
```

<img width="500" alt="history loss" src="https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/141728733/957af407-48d9-4af8-a3cb-8f039fcb48b2">

#

- Plot Graph
```python
import matplotlib.pyplot as  
plt.style.use('ggplot')

def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plot_history(history)
```
<img width="500" alt="history loss" src="https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/141728733/33f6d5f0-58d6-490e-a33a-0e65b3f577de">

#
- Testing predict the answer

```python 
import numpy as np
predict = model.predict(X_test)
predict
predicted_classes = np.argmax(predict, axis=1)
print(predicted_classes)
```
<details>
<summary><strong>Output</strong></summary>
<br>

```
372/372 [==============================] - 1s 2ms/step

[0 1 1 ... 2 2 0]
```
</details>

#

- Show the accuracy score
```python
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, predicted_classes)
score
```
<details>
<summary><strong>Output</strong></summary><br>

```
0.9955477150537635
```
</details>






