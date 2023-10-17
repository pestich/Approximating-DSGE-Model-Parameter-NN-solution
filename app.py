from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from predict_all import PredictAll
import tempfile


app = FastAPI()
just_do_it = PredictAll()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_file:
            file_data = await file.read()
            tmp_file.write(file_data)

    input_data = np.load(tmp_file.name)
    predictions = just_do_it(input_data)

    with tempfile.NamedTemporaryFile(delete=False, prefix='predictions_', suffix=".npy") as predictions_file:
        np.save(predictions_file, predictions)
        response_file = predictions_file.name
    return FileResponse(response_file, media_type="application/octet-stream")
