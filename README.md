# Neural Network for Approximating DSGE Model Parameters

## Project Description
This project is a neural network designed to approximate the parameters of an economic model simulator, DSGE, based on the output data of time series. The project uses a total of 6 neural networks, each of which predicts 15 model parameters, including the mean and quantiles.

## Neural Network Architecture
The neural network employs the following architecture:

```python
model = Sequential()

model.add(Conv2D(16, (3,3), activation='relu', input_shape=(time_steps, num_features, 1), padding='same'))
model.add(UpSampling2D((2,2)))
model add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, (12,6), activation='relu', strides=(6, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', strides=(2, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(15))

opt = AdamW(learning_rate=0.0001)
```

This architecture includes convolutional and fully connected layers, as well as upsampling and downsampling operations to process input time series data.

## Usage Instructions
To use this neural network for approximating the DSGE model, follow these steps:

1. Install the necessary libraries and dependencies, including FastAPI. You can use the `requirements.txt` file to install all the required dependencies. Run the following command in the project's root directory:
```python
pip install -r requirements.txt
```

2. Train the model from scratch using the available time series data.

3. Prepare data for use in the neural network. Ensure that the data conforms to the format specified in the `input_shape` parameters.

4. Deploy a FastAPI server using your neural network. Example code for deploying a FastAPI server can be found in the `app.py` file.
