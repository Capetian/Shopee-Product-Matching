
import os
import utils
import numpy as np
# import tensorflow as tf
from flask import Flask
from flask import Flask, render_template, request, redirect
from markupsafe import escape

# from utils import process_image, encode_features

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
image_handler = utils.ImageHandler()
text_handler = utils.TextHandler()
database = utils.DB()
# IMAGE_MODEL_PATH = '../train/models/0/'
# TABULAR_MODEL_PATH = '../train/models/tab_price_0/rfr_model.sav'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.filters['zip'] = zip

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
            
        photo = request.files['search_image']
        text = request.form['search_text']
        text = escape(text)

        # PREDICT BY PHOTO
        if photo.filename != '':
            # For now, If user passed photo, make predictions only using this photo
            error_msg = photo_is_valid(photo.filename)
            # if error_msg is not None:
            #     return render_template('error.html', error=error_msg)

            path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
            photo.save(path)

            img_ids = image_handler.run(path)
            image, title= database.query(img_ids)
            return render_template('predict.html', image=image, title=title)
        if  text != '':

            # if error_msg is not None:
            #     return render_template('error.html', error=error_msg)

            text_ids = text_handler.run(text)
            image, title= database.query(text_ids)
            return render_template('predict.html', image=image, title=title)

        # PREDICT BY TABULAR DATA
        # try:
        #     data_is_valid = validate_car_data(brand, mileage_kkm, fuel_type, transmission_type, year_made, engine_size)
        # except ValueError as err:
        #     return render_template('error.html', error=err)

        # if data_is_valid:
        #     prediction = predict_price(brand, mileage_kkm, fuel_type, transmission_type, year_made, engine_size)
        #     return render_template('predict.html', prediction=prediction)

    return render_template('index.html')


def photo_is_valid(filename):
    extension = filename.rsplit('.', 1)[-1].lower()
    if not (('.' in filename) and (extension in ALLOWED_EXTENSIONS)):
        error_msg = f"Invalid image extension: {extension}. Allowed extensions: { ', '.join(ALLOWED_EXTENSIONS)}"
        return error_msg
    return None


def validate_car_data(brand, mileage_kkm, fuel_type, transmission_type, year_made, engine_size):

    if mileage_kkm and int(mileage_kkm)< 0:
        raise ValueError('Kilometers run can not be negative')
    if engine_size and float(engine_size) < 0:
        raise ValueError('Engine size can not be negative')
    elif engine_size and float(engine_size) > 8.4:
        raise ValueError('Maximum engine size is 8.4 liters')
    return True


def predict_price(brand, mileage_kkm, fuel_type, transmission_type, year_made, engine_size):
    # data = encode_features(
    #     brand=brand,
    #     mileage_kkm=mileage_kkm,
    #     fuel_type=fuel_type,
    #     transmission_type=transmission_type,
    #     year_made=year_made,
    #     engine_size=engine_size,
    # )

    # model = joblib.load(TABULAR_MODEL_PATH)
    # prediction = int(model.predict(data)[0])
    # return prediction
    return 0


def predict_price_from_image(image):

    # # Additional setup may be needed to run with CPU or GPU
    # # Set CPU as available physical device
    # # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # # tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    # model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    # prediction = int(model.predict(image)[0, 0])
    # return prediction
    return 0


if __name__ == '__main__':
    app.run()

