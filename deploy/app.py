
import os
import utils
import numpy as np
from flask import Flask
from flask import Flask, render_template, request, redirect
from markupsafe import escape



UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
image_handler = utils.ImageHandler()
text_handler = utils.TextHandler()
database = utils.DB()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.filters['zip'] = zip

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
            
        photo = request.files['search_image']
        text = request.form['search_text']
        text = escape(text)
        img_ids = None
        text_ids = None
        # PREDICT BY PHOTO
        if photo.filename != '':
            
            error_msg = photo_is_valid(photo.filename)
            if error_msg is not None:
                return render_template('error.html', error=error_msg)

            path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
            photo.save(path)

            img_ids = image_handler.run(path)

        if  text != '':
            text_ids = text_handler.run(text)

        if text_ids is not None and img_ids is not None:
            combined_ids = utils.combine_ids(img_ids, text_ids)
            image, title= database.query(combined_ids)
            return render_template('predict.html', image=image, title=title)
        elif text_ids is not None:
            image, title= database.query(text_ids)
            return render_template('predict.html', image=image, title=title)
        elif img_ids is not None:
            image, title= database.query(img_ids)
            return render_template('predict.html', image=image, title=title)


    return render_template('index.html')


def photo_is_valid(filename):
    extension = filename.rsplit('.', 1)[-1].lower()
    if not (('.' in filename) and (extension in ALLOWED_EXTENSIONS)):
        error_msg = f"Invalid image extension: {extension}. Allowed extensions: { ', '.join(ALLOWED_EXTENSIONS)}"
        return error_msg
    return None



if __name__ == '__main__':
    app.run()

