from flask import Flask, render_template, request, send_file
from PIL import Image
import os

app = Flask(__name__)

def superres(input_image):
    # TODO: Actual SuperRes after training.
    output_image = input_image.resize((input_image.width * 2, input_image.height * 2))
    return output_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return render_template('index.html', error='Invalid file format')

        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        input_image = Image.open(file_path)
        output_image = superres(input_image)

        output_folder = 'output'
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'output.png')
        output_image.save(output_path)

        return send_file(output_path, as_attachment=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
