from flask import Flask, redirect, url_for, request, render_template
from image_helper import ImageHelper
from gan import USERGAN
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from tensorflow import get_default_graph, Session
import numpy as np

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



'''
initialise model
'''
sess = Session()
graph = get_default_graph()
set_session(sess)
input_discriminator = load_model("models/dcgan_discriminator.h5")
input_generator = load_model("models/dcgan_generator.h5")

image_helper = ImageHelper()
user_gan = USERGAN(
            image_shape=(28, 28, 1), generator_input_dim=100,
            image_helper=image_helper, img_channels=1,
            pre_load_discriminator=input_discriminator,
            pre_load_generator=input_generator
            )



'''
app setup
'''
@app.route('/init', methods=['GET','POST'])
def init():
    number_of_images = int(request.form['number_of_images_per_iteration'])
    
    return redirect(url_for('images', number_of_images=number_of_images))



@app.route('/images/<int:number_of_images>')
def images(number_of_images):    
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        user_gan.generate_images(number_of_images, 1,
                                 'static/image_directory/display_directory',
                                 'static/image_directory/permanent_directory')
    
    return render_template('display_images.html')



@app.route('/retrain', methods=['POST'])
def retrain():
    user_response = request.form
    augmentations = int(request.form['augmentations_per_image'])
    
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        img_for_retrain, lbl_for_retrain = user_gan.process_images(
                'static/image_directory/display_directory',
                user_response
                )
        user_gan.retrain(img_for_retrain, lbl_for_retrain,
                         augmentations, 32)
    
    return redirect(url_for('images', number_of_images=len(lbl_for_retrain)))



@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response



if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000) # host 0.0.0.0 makes available externally, port defaults to 5000





