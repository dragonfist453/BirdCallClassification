from flask import Flask, flash, request, redirect, url_for, send_from_directory
import os

from flask.templating import render_template
from nn import *
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.getcwd() + '\\backend\\input\\'
OUTPUT_FOLDER = os.getcwd() + '\\backend\\output\\'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

bird_names = ['WCB', 'CBS', 'RWB', 'NFP', 'ISB']

app = Flask(__name__, static_folder= os.getcwd() + '\\backend')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = build_network()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict_bird', input=filename))
    return '''
    <!doctype html>
    <title>Bird Call Classification</title>
    <h1>Upload audio file</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<name>', methods=['GET'])
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/predict_bird/<input>', methods=['GET'])
def predict_bird(input):
    imgDims = (224,224)
    output = []
    audioToMel(input, UPLOAD_FOLDER, OUTPUT_FOLDER)
    image = tf.keras.preprocessing.image.load_img(OUTPUT_FOLDER + input[:-4] + '.png', target_size=imgDims)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    result = model.predict(input_arr)
    output = [bird_names[ele] for ele in np.argmax(result, axis=1)]
    print(output)
    return redirect(url_for('show_bird', code=output[0]))

@app.route('/bird/<code>', methods=['GET'])
def show_bird(code):
    print(code)
    return render_template('bird.html')

if __name__=='__main__':
    app.run(port=8000)