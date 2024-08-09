# Simple Neural Network Image Classifier

This repository contains a simple Convolutional Neural Network (CNN) implemented using TensorFlow and Keras. The model classifies images of cats and dogs. The dataset includes training and testing images of cats and dogs.

## Project Structure

- `LICENSE`: License file.
- `README.md`: Project documentation.
- `Simple Neural Network.ipynb`: Jupyter Notebook containing the model and training code.
- `dataset/`: Directory containing the training and testing datasets.
  - `training_set/`: Directory containing training images.
  - `testing_set/`: Directory containing testing images.
  - `single_prediction/`: Directory containing a single image for prediction.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Jupyter Notebook

You can install the necessary packages using pip:

```bash
pip install tensorflow keras numpy
```
## How to Use

### Data Preprocessing:

- The training and testing datasets are preprocessed using `ImageDataGenerator` from Keras.
- Images are rescaled, and the training images are augmented with random transformations like shear and zoom.

### Model Building:

- A simple CNN is built with two convolutional layers, each followed by a max-pooling layer.
- The model is compiled using the Adam optimizer and binary cross-entropy loss function.
- The model is trained for 25 epochs on the training dataset and validated on the test dataset.

### Prediction:

- The trained model can predict whether a new image is of a cat or a dog.
- Simply place an image in the `dataset/single_prediction/` directory and run the prediction code.

### Example Usage

```python
# Load and preprocess the image
test_image = tf.keras.utils.load_img("dataset/single_prediction/catordog.jpg", target_size=(64, 64))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict using the trained model
result = cnn.predict(test_image)

# Output the prediction
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print("Prediction:", prediction)
```
## Results

The model achieves an accuracy of around 94% on the training set. However, the accuracy on the test set may vary depending on the dataset used.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed using TensorFlow and Keras. Thanks to the developers of these libraries for providing such powerful tools for deep learning.

