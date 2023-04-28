from flask import Flask, url_for, render_template, request, jsonify
from torchvision import transforms as tfs
import PIL.ImageOps
import numpy as np
from skimage import measure

import sys
sys.path.insert(0,'../.')


from utils import Generator as netG
import torch.nn as nn
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


available_device = 'cpu'

transformations = [
                   tfs.Grayscale(),
                   tfs.Resize((128, 128)),
                   #tfs.Lambda(lambda x: PIL.ImageOps.invert(x)),
                   tfs.ToTensor()
]

trans = tfs.Compose(transformations)


# Model Initiaimgsation
model_G = netG.Generator(nc_input=2, nc_output=1).to(available_device)
checkpoint = torch.load("../models/generador_v9_current_5000.pkl", map_location=torch.device('cpu'))
model_G.load_state_dict(checkpoint)
model_G = model_G.eval()

app = Flask(__name__)
with app.test_request_context():
    url_for('static', filename='js/init.js')
    url_for('static', filename='js/paint.js')
    url_for('static', filename='image/CC_02_7.png')
    url_for('static', filename='image/MB_14.png')


@app.route("/")
def paint():
    return render_template('paint.html')


from PIL import Image
from io import BytesIO
import base64

import cv2

@app.route('/sendImage', methods=['POST'])
def sendImage():
    if request.method == 'POST':
        
        im = Image.open(BytesIO(base64.b64decode(request.form['files'].replace('data:image/png;base64',''))))
        background = Image.new('RGBA', im.size, (255,255,255))
        alpha_composite = Image.alpha_composite(background, im)
        alpha_composite = alpha_composite.convert('RGB')
        alpha_composite.save('static/image/uploaded_file.jpg', 'JPEG' )
        
        #image = trans(Image.open('/var/www/uploaded_file.jpg')).reshape(-1, 1, 128, 128)
        image = trans(alpha_composite).reshape(-1, 1, 128, 128)
        image = image.to(available_device)
        blacks = torch.zeros_like(image).to(available_device)
        origin_A = torch.cat((image, blacks), 1)
        predicted_A = model_G(origin_A)
        predicted_A = predicted_A + image
        predicted_A[predicted_A>1] = 1 
        result = predicted_A.detach().cpu().numpy().reshape(128, 128)
        result = (1 - result) * 255
        output = cv2.resize(result, (512, 512))
        cv2.imwrite('static/image/result.png', output)
        img_f_2d = output
        img_s = np.ones((img_f_2d.shape[0] + 100, img_f_2d.shape[0] + 100)) * 255
        img_s[50:-50, 50:-50] = img_f_2d
        img_f_2d = img_s
        contours = measure.find_contours(img_f_2d, 128)

        contour  = np.array(contours)[0]

        try:
            contour = interpcurve(100,  contour[:, 0],  contour[:, 1])
        except:
            pass

        plt.figure()
        plt.fill(contour[:, 1] , contour[:, 0] * -1, 'k')
        plt.axes().set_aspect('equal')
        plt.axis('off')
        plt.savefig('static/image/result_land.png')


        resp = jsonify(success=True)
        return resp

'''
@app.route('/getResult', methods=['POST'])
def getResult():
    if request.method == 'GET':
        
        im = Image.open('static/image/result.png')
        img
        resp = jsonify(success=True)
        return resp
'''


if __name__=='__main__':
    app.run(debug=True, port=6789)
    
#@app.errorhandler(404)
#def page_not_found(error):
#    return render_template('page_not_found.html'), 404