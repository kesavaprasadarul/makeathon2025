from fastapi import FastAPI, UploadFile, File, Form
import pydantic
from fastapi import APIRouter, Depends, HTTPException
import os
import fastapi
import zipfile
from io import BytesIO
app = FastAPI()

class DataSource:
    def __init__(self, name, folderpath, images_folder_path):
        self.name = name
        self.set_folderpath(folderpath)
        self.set_images_folder_path(images_folder_path)

    def set_folderpath(self, folderpath):
        self.folderpath = folderpath
        self.camera_json_path = os.path.join(folderpath, "cameras.json")
        self.coords_path = os.path.join(folderpath, "odm_georeferencing/coords.txt")
        self.dtm_path = os.path.join(folderpath, "odm_dem/dtm.tif")
    
    def set_images_folder_path(self, images_folder_path):
        self.images_folder_path = images_folder_path
        self.image_paths = [os.path.join(images_folder_path, f) for f in os.listdir(images_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

regensburg = DataSource(name ="regensburg", 
                        folderpath = "/datasets/nodeodm_out_4",
                        images_folder_path ="/datasets/images/regensburg")

munich = DataSource(name ="munich",
                    folderpath = "/datasets/nodeodm_test_1",
                    images_folder_path ="/datasets/images/munich")

@app.post("/getImageByIndex")
async def get_image(data_source: str, image_index: int):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    if image_index < 0 or image_index >= len(data_source.image_paths):
        raise HTTPException(status_code=400, detail="Invalid image index")

    image_path = data_source.image_paths[image_index]
    return fastapi.responses.FileResponse(image_path, media_type="image/jpeg")

@app.post("/getImageByName")
async def get_image_by_name(data_source: str, image_name: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    image_path = os.path.join(data_source.images_folder_path, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return fastapi.responses.FileResponse(image_path, media_type="image/jpeg")

@app.post("/getCameraJson")
async def get_camera_json(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    return fastapi.responses.FileResponse(data_source.camera_json_path, media_type="application/json")

@app.post("/getCoords")
async def get_coords(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    return fastapi.responses.FileResponse(data_source.coords_path, media_type="text/plain")

@app.post("/getDtm")
async def get_dtm(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    return fastapi.responses.FileResponse(data_source.dtm_path, media_type="image/tiff")

@app.post("/getLas")
async def get_las(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    las_path = os.path.join(data_source.folderpath, "odm_georeferencing/odm_georeferenced_model.laz")
    if not os.path.exists(las_path):
        raise HTTPException(status_code=404, detail="LAS file not found")

    return fastapi.responses.FileResponse(las_path, media_type="application/octet-stream")


@app.post("/getGpkg")
async def get_las(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    las_path = os.path.join(data_source.folderpath, "odm_georeferencing/odm_georeferenced_model.bounds.gpkg")
    if not os.path.exists(las_path):
        raise HTTPException(status_code=404, detail="LAS file not found")

    return fastapi.responses.FileResponse(las_path, media_type="application/octet-stream")

@app.get("/getProjectionConversionString")
async def get_projection_conversion_string(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    with open(os.path.join(data_source.folderpath, "odm_georeferencing/proj.txt"), 'r') as f:
        lines = f.readlines()
        conversion_string = lines[0].strip()
        return {"conversion_string": conversion_string}

@app.get("/getObjFile")
async def get_obj_file(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    obj_path = os.path.join(data_source.folderpath, "odm_texturing/textured_model.obj")
    # Package all .glb, .conf, and .mtl files in that folder, zip and return the zip file

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        folder_path = os.path.join(data_source.folderpath, "odm_texturing")
        for file_name in os.listdir(folder_path):
            if file_name.endswith((".glb", ".conf", ".mtl")):
                file_path = os.path.join(folder_path, file_name)
                zip_file.write(file_path, arcname=file_name)

    zip_buffer.seek(0)
    return fastapi.responses.StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={data_source.name}_mesh_files.zip"}
    )


@app.get("/getPointCloud")
async def get_point_cloud(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    # point cloud files are in 3d_tiles\pointcloud, get all files in that folder, zip and return
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        folder_path = os.path.join(data_source.folderpath, "3d_tiles/pointcloud")
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            zip_file.write(file_path, arcname=file_name)
    zip_buffer.seek(0)
    return fastapi.responses.StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={data_source.name}_pc_files.zip"}
    )
    

@app.post("/getImagePaths")
async def get_image_paths(data_source: str):
    if data_source == "regensburg":
        data_source = regensburg
    elif data_source == "munich":
        data_source = munich
    else:
        raise HTTPException(status_code=400, detail="Invalid data source")

    return {"image_paths": data_source.image_paths}