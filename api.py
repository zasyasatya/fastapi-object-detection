from fastapi import FastAPI, File, UploadFile, Request, status, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import supervision as sv
import numpy as np
import cv2
import os
import time
from functools import wraps

app = FastAPI()

'''
    Allow CORS
'''
origins = [
    "http://localhost:3000",  # React
    "http://localhost:8080",  # Vue.js
    "http://localhost:8000",  # Angular
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


imageDirectory = "uploadedFile" # store uploaded image in this folder

if not os.path.exists(imageDirectory):
    os.makedirs(imageDirectory)

model = YOLO("dataset_v2_hand-sign.pt") #

def rate_limited(max_calls: int, time_frame: int):
    def decorator(func):
        calls = []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            calls_in_time_frame = [call for call in calls if call > now - time_frame]
            if len(calls_in_time_frame) >= max_calls:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded!")
            calls.append(now)
            return await func(*args, *kwargs)

        return wrapper

    return decorator


def objectDetector(filename):
    print("------------------------", filename)
    frame = cv2.imread("uploadedFile/" + filename)
    results = model(frame, conf=0.7)[0]

    detections = sv.Detections.from_yolov8(results)
    results = detections.with_nms(threshold=0.5)
    class_name = ""
    confidence = 0.0
    bbox = results.xyxy # get box coordinates in (top, left, bottom, right) format
    bbox_class = results.class_id # cls, (N, )
    for r in results:
        '''
            index 0: bounding box
            index 1: mask
            index 2: confidence
            index 3: class_id
            index 4: tracker_id
        '''
        frame = np.ascontiguousarray(frame)
        annotator = Annotator(frame)
        box = r[0]
        confidence = r[2]
        class_id = r[3]

        # get class name
        class_name = model.names[int(class_id)]

        # draw label
        annotator.box_label(box, class_name + ' ' + str(int(confidence * 100)) + '%', color=(0, 0, 255), txt_color=(255, 255, 255))
        frame = annotator.result()

    jsonResult = {
        "status": "error"
    }

    bbox_json = {}
    cv2.imwrite("result.png", frame)
    if class_name is not None and class_name != "":
        print(f"=====CLASSNAME===={class_name}")
        print(f"===========INI BBOX: {bbox}")
        if bbox is not None and len(bbox) > 0:
            x1, y1, x2, y2 = bbox[0]
            centroid_x = float((x1 + x2) / 2) 
            centroid_y = float((y1 + y2) / 2) 

            bbox_json = {
                "x1": float(x1), 
                "y1": float(y1), 
                "x2": float(x2), 
                "y2": float(y2), 
                "centroid_x": centroid_x,
                "centroid_y": centroid_y
            }

        jsonResult = {
            "status": "successful",
            "bbox": bbox_json,
            "class_name": class_name,
            "confidence": float(confidence),
             "path": "result.png",
        }

    JSONResponse(jsonResult)


@app.get("/")
@rate_limited(max_calls=100, time_frame=60) # decorator to limit request
async def index():
    return {"message": "Hellow World"}


@app.post("/upload")
#@rate_limited(max_calls=100, time_frame=60) # decorator to limit request
async def uploadFile(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    #save the file
    with open(f"{imageDirectory}/{file.filename}", "wb") as f:
        f.write(contents)

    detectionResult = objectDetector(file.filename)
    print("============================", detectionResult)
    return JSONResponse(detectionResult)


@app.get("/detectedImage")
# @rate_limited(max_calls=100, time_frame=60) # decorator to limit request
async def showImage():
    if (os.path.exists("result.png")):
        imagePath = "result.png"
        return FileResponse(imagePath)
    else:
        return {"status", "error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info") # adjust port 



