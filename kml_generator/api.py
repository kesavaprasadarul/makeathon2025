from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
import xml.etree.ElementTree as ET
import xml.dom.minidom
import zipfile
import io
import time

# Constants
DEFAULT_ALTITUDE = 20.0  # meters AGL
KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
WPML_NAMESPACE = "http://www.dji.com/wpmz/1.0.6"

ET.register_namespace('', KML_NAMESPACE)
ET.register_namespace('wpml', WPML_NAMESPACE)

app = FastAPI(
    title="KML/KMZ Generator",
    description="API to generate KML or KMZ files from GPS points for DJI waypoint missions.",
    version="1.0.0"
)

class Point(BaseModel):
    latitude: float = Field(..., description="Latitude of the waypoint")
    longitude: float = Field(..., description="Longitude of the waypoint")
    altitude: float = Field(DEFAULT_ALTITUDE, description="Height AGL in meters")

class MissionRequest(BaseModel):
    points: List[Point]
    author: str = Field("API User", description="Mission creator name")
    global_altitude: float = Field(20.0, description="Global flight height (m)")
    speed: float = Field(5.0, description="Global flight speed (m/s)")

@app.post("/generate_kml")
def generate_kml(request: MissionRequest):
    if not request.points:
        raise HTTPException(status_code=400, detail="Point list cannot be empty.")

    # Build KML structure
    kml = ET.Element('kml')
    document = ET.SubElement(kml, 'Document')

    # Metadata
    ET.SubElement(document, f'{{{WPML_NAMESPACE}}}author').text = request.author
    timestamp = str(int(time.time() * 1000))
    ET.SubElement(document, f'{{{WPML_NAMESPACE}}}createTime').text = timestamp
    ET.SubElement(document, f'{{{WPML_NAMESPACE}}}updateTime').text = timestamp

    folder = ET.SubElement(document, 'Folder')

    # Waypoints
    for idx, pt in enumerate(request.points):
        placemark = ET.SubElement(folder, 'Placemark')
        point = ET.SubElement(placemark, 'Point')
        ET.SubElement(point, 'coordinates').text = f"{pt.longitude},{pt.latitude},{pt.altitude}"

        # DJI-specific tags
        ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}index').text = str(idx)
        ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}ellipsoidHeight').text = str(pt.altitude)
        ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}height').text = str(request.global_altitude)
        ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}waypointSpeed').text = str(request.speed)

    # Pretty print
    rough = ET.tostring(kml, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough)
    kml_str = reparsed.toprettyxml(indent="  ")

    # Prepare KMZ in-memory
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('mission.kml', kml_str)
    buffer.seek(0)

    headers = {
        'Content-Disposition': 'attachment; filename=mission.kmz'
    }
    return StreamingResponse(buffer, media_type='application/vnd.google-earth.kmz', headers=headers)

@app.get("/health")
def health_check():
    return {"status": "ok"}
