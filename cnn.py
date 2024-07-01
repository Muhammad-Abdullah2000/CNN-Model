import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



# Load and preprocess the image
image_path = "mabdullah.png"
image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
input_arr = keras.preprocessing.image.img_to_array(image)
input_arr = tf.expand_dims(input_arr, 0)
input_arr = keras.applications.vgg16.preprocess_input(input_arr)

# Load the pre-trained VGG16 model
model = keras.applications.VGG16(weights="imagenet", include_top=True)

# Perform prediction on the image
predictions = model.predict(input_arr)
decoded_predictions = keras.applications.vgg16.decode_predictions(predictions, top=3)

# Print the predicted labels
for pred in decoded_predictions[0]:
    print(pred[1], pred[2])