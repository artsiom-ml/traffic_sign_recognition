from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

import cv2
import numpy as np

import torch
import base64
import random

app = FastAPI()
templates = Jinja2Templates(directory='templates')

model_selection_options = ['yolov5s6', 'yolov5s6_1', 'yolov5s']
model_dict = {model_name: None for model_name in model_selection_options}


colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]

def get_yolov5(model_name):
    model = torch.hub.load("./../yolov5", 'custom', path=f'./../models/{model_name}.pt', source='local')
    return model

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse('home.html', {
            "request": request,
            "model_selection_options": model_selection_options,
        })

@app.get("/drag_and_drop_detect")
async def drag_and_drop_detect(request: Request):
    
    return templates.TemplateResponse('drag_and_drop_detect.html',
            {"request": request,
            "model_selection_options": model_selection_options,
        })


@app.post("/")
async def detect_with_server_side_rendering(request: Request,
                        file_list: List[UploadFile] = File(...), 
                        model_name: str = Form(...),
                        img_size: int = Form(1280)):
    
    if model_dict[model_name] is None:
        model_dict[model_name] = get_yolov5(model_name)
        
    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                    for file in file_list]

    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    results = model_dict[model_name](img_batch_rgb, size=img_size)

    json_results = results_to_json(results, model_dict[model_name])

    img_str_list = []
    #plot bboxes on the image
    for img, bbox_list in zip(img_batch, json_results):
        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], img, label=label, 
                    color=colors[int(bbox['class'])], line_thickness=2)

        img_str_list.append(base64EncodeImage(img))

    encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')
    return templates.TemplateResponse('show_results.html', {
            'request': request,
            'bbox_image_data_zipped': zip(img_str_list,json_results),
            'bbox_data_str': encoded_json_results,
        })


@app.post("/detect")
async def detect_via_api(file_list: List[UploadFile] = File(...), 
                model_name: str = Form(...),
                img_size: Optional[int] = Form(1280),
                download_image: Optional[bool] = Form(False)):
    
    if model_dict[model_name] is None:
        model_dict[model_name] = get_yolov5(model_name)
    
    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                for file in file_list]

    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]
    
    results = model_dict[model_name](img_batch_rgb, size = img_size) 
    json_results = results_to_json(results,model_dict[model_name])

    if download_image:
        for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
            for bbox in bbox_list:
                label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                plot_one_box(bbox['bbox'], img, label=label, 
                        color=colors[int(bbox['class'])], line_thickness=2)

            payload = {'image_base64': base64EncodeImage(img)}
            json_results[idx].append(payload)

    return json_results


@app.get("/video_feed/{camera_id}", include_in_schema=False)
async def video_feed(camera_id: int):
    return StreamingResponse(gen_frames(camera_id), media_type='multipart/x-mixed-replace; boundary=frame')


async def gen_frames(camera_id):
    cap=  cv2.VideoCapture(camera_id)

    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 



def results_to_json(results, model):
    return [
                [
                    {
                    "class": int(pred[5]),
                    "class_name": model.model.names[int(pred[5])],
                    "bbox": [int(x) for x in pred[:4].tolist()],
                    "confidence": float(pred[4]),
                    }
                for pred in result
                ]
            for result in results.xyxy
    ]

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl-1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def base64EncodeImage(img):
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

    return im_b64

if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    parser.add_argument('--precache-models', action='store_true', 
            help='Pre-cache all models in memory upon initialization, otherwise dynamically caches models')
    opt = parser.parse_args()

    if opt.precache_models:
        model_dict = {model_name: get_yolov5(model_name)
                        for model_name in model_selection_options}

    app_str = 'main:app'
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=False)
