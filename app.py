from flask import Flask, render_template, request
import cv2
import numpy as np  # Add this import
import matplotlib.pyplot as plt
# import image_dehazer
import os
from werkzeug.utils import secure_filename
import skimage
from skimage.restoration import (denoise_wavelet,estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#CLAHE

def enhance_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(20, 20))
    v = clahe.apply(v)

    hsv_img = np.dstack((h, s, v))
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rgb


#dark channel prior, transmission map, atmospheric

def dehaze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    dehazed, _ = image_dehazer.remove_haze(img)

    return cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)


#bicubic interpolation

def enhance_resolution(image_path, scale_factor=1.5):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Upscale the image using bicubic interpolation
    image_hr = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply image enhancement technique 
    image = cv2.detailEnhance(image_hr)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ... (previous code)

#CLAHE

def enhance_low_light(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Increase brightness and contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 100    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Convert the adjusted image to LAB color space
    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)

    # Split the LAB image into separate channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge the enhanced L channel with the original A and B channels
    lab_enhanced = cv2.merge((l_enhanced, a, b))

    # Convert the enhanced LAB image back to RGB color space
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)




#guassian high pass filter

def sharpen_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Apply Gaussian Blur to the image
    blur = cv2.GaussianBlur(img, (7, 7), 2)

    # Add the high-pass filter back to the original image to get the sharpened image
    sharpened = cv2.addWeighted(img, 4.5, blur, -3.5, 0)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

# ... (previous code)

@app.route('/sharpen', methods=['GET', 'POST'])
def sharpen():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('sharpen.html', result_image=None, error="No file selected.")

        file = request.files['image']

        if file.filename == '':
            return render_template('sharpen.html', result_image=None, error="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            original_img, sharpened_img = sharpen_image(filepath)

            if sharpened_img is not None:
                plt.subplot(2, 2, 1), plt.imshow(original_img), plt.title('Original Image')
                plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 2), plt.imshow(sharpened_img), plt.title('Sharpened Image')
                plt.xticks([]), plt.yticks([])
                plt.subplots_adjust(left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.1, hspace=0.1)
                plt.savefig('static/sharpened_image.png', bbox_inches='tight')
                plt.clf()

                return render_template('sharpen.html', result_image='static/sharpened_image.png', error=None)

    return render_template('sharpen.html', result_image=None, error=None)

# ... (previous code)


@app.route('/enhance_low_light', methods=['GET', 'POST'])
def enhance_low_light_route():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('enhance_low_light.html', result_image=None, error="No file selected.")

        file = request.files['image']

        if file.filename == '':
            return render_template('enhance_low_light.html', result_image=None, error="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            original_img, enhanced_img = enhance_low_light(filepath)

            if enhanced_img is not None:
                plt.subplot(2, 2, 1), plt.imshow(original_img), plt.title('Original Image')
                plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 2), plt.imshow(enhanced_img), plt.title('Enhanced Low-Light Image')
                plt.xticks([]), plt.yticks([])
                plt.subplots_adjust(left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.1, hspace=0.1)
                plt.savefig('static/enhanced_low_light_image.png', bbox_inches='tight')
                plt.clf()

                return render_template('enhance_low_light.html', result_image='static/enhanced_low_light_image.png', error=None)

    return render_template('enhance_low_light.html', result_image=None, error=None)

# ... (previous code)




@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image_enhancement', methods=['GET', 'POST'])
def image_enhancement():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('image_enhancement.html', result_image=None, error="No file selected.")

        file = request.files['image']

        if file.filename == '':
            return render_template('image_enhancement.html', result_image=None, error="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            original_img, enhanced_img = enhance_image(filepath)

            if original_img is not None:
                plt.subplot(1, 2, 1)
                plt.imshow(original_img)
                plt.xticks([]), plt.yticks([])
                plt.title("Original Image", fontsize=10)

                plt.subplot(1, 2, 2)
                plt.imshow(enhanced_img)
                plt.xticks([]), plt.yticks([])
                plt.title("Enhanced Image", fontsize=10)

                plt.colorbar()
                plt.savefig('static/output_image.png')
                plt.clf()

                return render_template('image_enhancement.html', result_image='static/output_image.png', error=None)

    return render_template('image_enhancement.html', result_image=None, error=None)


@app.route('/dehaze', methods=['GET', 'POST'])
def dehaze():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('dehaze.html', result_image=None, error="No file selected.")

        file = request.files['image']

        if file.filename == '':
            return render_template('dehaze.html', result_image=None, error="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            dehazed_img = dehaze_image(filepath)

            if dehazed_img is not None:
                fig, ax = plt.subplots()
                ax.imshow(dehazed_img)
                ax.axis('off')
                plt.savefig('static/dehazed_image.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig)  # Close the figure to release resources
                return render_template('dehaze.html', result_image='static/dehazed_image.png', error=None)

    return render_template('dehaze.html', result_image=None, error=None)

@app.route('/enhance_resolution', methods=['GET', 'POST'])
def enhance_resolution_route():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('enhance_resolution.html', result_image=None, error="No file selected.")

        file = request.files['image']

        if file.filename == '':
            return render_template('enhance_resolution.html', result_image=None, error="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            original_img, enhanced_img = enhance_resolution(filepath)

            if enhanced_img is not None:
                plt.subplot(2, 2, 1), plt.imshow(original_img), plt.title('Original Image')
                plt.xticks([]), plt.yticks([])
                plt.subplot(2, 2, 2), plt.imshow(enhanced_img), plt.title('Enhanced Resolution Image')
                plt.xticks([]), plt.yticks([])
                plt.subplots_adjust(left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.1, hspace=0.1)
                plt.savefig('static/enhanced_resolution_image.png', bbox_inches='tight')
                plt.clf()

                return render_template('enhance_resolution.html', result_image='static/enhanced_resolution_image.png', error=None)

    return render_template('enhance_resolution.html', result_image=None, error=None)








if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
