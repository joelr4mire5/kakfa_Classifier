from flask import Flask, Response
from kafka import KafkaConsumer
import os
import keras
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


consumer = KafkaConsumer('fight_moves', bootstrap_servers='localhost:9092')
""" model = keras.models.load_model('save_at_1.keras')
 """
app = Flask(__name__)


# Folder to save the images
image_folder = 'New'
os.makedirs(image_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Counter for naming the images
image_counter = 0

def save_image(image_data):
    global image_counter
    image_counter += 1
    image_filename = os.path.join(image_folder, f'image_{image_counter}.jpg')
    with open(image_filename, 'wb') as f:
        f.write(image_data)
    print(f'Saved image: {image_filename}')


def kafkastream():
    for message in consumer:
        image_data = message.value

        save_image(image_data)
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + message.value + b'\r\n\r\n')
        
        """ try:
            
            
            img_path = os.path.join(image_folder, f'image_{image_counter}.jpg')
            img = keras.utils.load_img(img_path, target_size=(180, 180))
            img_array = keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array)
            score = float(predictions[0])
            print(f"This image is {100 * (1 - score):.2f}% Bakat an maki and {100 * score:.2f}% Monton.")


            


        except Exception as e:
            print("Error during prediction",str(e)) """
            


        



""" def kafkastream():
    for message in consumer:
        image_data = message.value
        save_image(image_data)  # Save the image to the folder
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + message.value + b'\r\n\r\n')
        message
  """


@app.route('/')
def index():
    return Response(kafkastream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
