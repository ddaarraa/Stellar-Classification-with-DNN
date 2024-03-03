# About Dataset :
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


# Data Visualization


![Unknown](https://github.com/640710505/Sellar-Classification-using-Neural-Network/assets/114089025/e5c9f836-24ab-4d1e-b257-d0c5f782256a)
#
![Unknown](https://github.com/640710505/Sellar-Classification-using-Neural-Network/assets/114089025/dfadc634-c486-4f6f-9fa8-3cad6e3d14b0)
#
![Unknown](https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/114089025/9c0a3c05-6f9e-453e-a80d-01a776fe7405)
#




# Feature important 
### With Logistic Regression

```python
### All feature and Target(class)  
X = data[['obj_ID','alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID',
       'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'redshift',
       'plate', 'MJD', 'fiber_ID']]
Y = data[['class']]
```
```python
###  use train_test_split  by train 70 % and test 30%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, 
                                                 random_state=42,
                                                 shuffle 
                                                 = True,stratify= Y)
```
```python
### Use StandardScaler for scaling data to samesame  value 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test) 
```
```python
### test Model
model = LogisticRegression()
model.fit(X_train_std,Y_train)
```
```python
model.score(X_test_std,Y_test)
```
<details>
<summary><strong>Output</strong></summary>

```
0.948473816858023
```
</details>

```python
print(coefficients)
```
<details>
  <summary><strong>Output</strong></summary>

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

## Coefficient Graph
![featureimportant](https://github.com/640710505/Sellar-Classification-using-Neural-Network/assets/141728733/10905af0-4587-4fcf-bc27-d759f84fc7d1)



# Feature Extraction
We should only important feature!!
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









# Neural Network Model

```python
X = dataset[['u', 'g', 'r', 'i', 'z', 'redshift','distance', 'NewAMagnitude']]
y = dataset[['class']]
```
```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, 
                                                     shuffle=True)

# one hot encoding of y
y_train_one_hot = to_categorical(y_train, num_classes=3)
y_test_one_hot = to_categorical(y_test, num_classes=3)
print(y_train_one_hot.shape)
```
<details>
<summary><strong>Output</strong></summary>

```
(47612, 3)
```
</details>



```python
from keras import Sequential
from keras.layers import Dense
from keras.backend import clear_session

model = Sequential()
model.add(Dense(units=2000, activation='relu', input_shape=(8,)))
model.add(Dense(units=1000 , activation='relu'))
model.add(Dense(units=100 , activation='relu'))
model.add(Dense(units= 10 , activation='relu'))
model.add(Dense(units=3, activation='sigmoid'))
model.summary()
```
<img width="500" alt="Model sequential" src="https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/141728733/c967ad6f-ae21-44b0-b58e-c90d596f6e87">


```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(optimizer="adam", loss='categorical_crossentropy')
history = model.fit(X_train, y_train_one_hot, epochs=9, batch_size = 60, validation_data=(X_test, y_test_one_hot))
```

<img width="500" alt="history loss" src="https://github.com/640710505/Stellar-Classification-using-Neural-Network/assets/141728733/957af407-48d9-4af8-a3cb-8f039fcb48b2">

```python
#function plot graph ref:notebookอาจารย์
import matplotlib.pyplot as plt
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

```python
import numpy as np
predict = model.predict(X_test)
predict
predicted_classes = np.argmax(predict, axis=1)
print(predicted_classes)
```
<details>
<summary><strong>Output</strong></summary>

```
372/372 [==============================] - 1s 2ms/step

[0 1 1 ... 2 2 0]
```
</details>




```python
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, predicted_classes)
score
```
<details>
<summary><strong>Output</strong></summary>

```
0.9955477150537635
```
</details>



# Score Analysis
# 


