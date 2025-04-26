import pandas as pd
import xml.etree.ElementTree as ET
import xml.dom.minidom
import zipfile
import os
import time

# Configurable parameters
DEFAULT_ALTITUDE = 20.0  # Default height AGL (meters) if missing
DEFAULT_SPEED = 5.0      # Default speed (m/s)
DEFAULT_TAKEOFF_SECURITY_HEIGHT = 10  # meters
DEFAULT_GLOBAL_RTH_HEIGHT = 100  # meters
DEFAULT_TRANSITIONAL_SPEED = 15  # m/s
DEFAULT_GLOBAL_HEIGHT = 20  # m
DEFAULT_AUTHOR = "Peter Schiekofer"

# Output files
OUTPUT_KML = 'mission.kml'
OUTPUT_KMZ = 'mission.kmz'

# Read CSV
df = pd.read_csv('coordinates.csv')

# Fill missing altitudes
df['altitude'] = df['altitude'].fillna(method='ffill').fillna(DEFAULT_ALTITUDE)

# Setup namespaces
KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
WPML_NAMESPACE = "http://www.dji.com/wpmz/1.0.6"
ET.register_namespace('', KML_NAMESPACE)
ET.register_namespace('wpml', WPML_NAMESPACE)

# Start building KML
kml = ET.Element('kml')
document = ET.SubElement(kml, 'Document')

# Add metadata
ET.SubElement(document, f'{{{WPML_NAMESPACE}}}author').text = DEFAULT_AUTHOR
ET.SubElement(document, f'{{{WPML_NAMESPACE}}}createTime').text = str(int(time.time() * 1000))
ET.SubElement(document, f'{{{WPML_NAMESPACE}}}updateTime').text = str(int(time.time() * 1000) + 10000)

# Add mission config
mission_config = ET.SubElement(document, f'{{{WPML_NAMESPACE}}}missionConfig')
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}flyToWaylineMode').text = 'safely'
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}finishAction').text = 'goHome'
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}exitOnRCLost').text = 'goContinue'
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}executeRCLostAction').text = 'goBack'
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}takeOffSecurityHeight').text = str(DEFAULT_TAKEOFF_SECURITY_HEIGHT)

first_point = df.iloc[0]
takeoff_ref = f"{first_point['latitude']},{first_point['longitude']},{first_point['altitude']}"
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}takeOffRefPoint').text = takeoff_ref
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}takeOffRefPointAGLHeight').text = '0'
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}globalTransitionalSpeed').text = str(DEFAULT_TRANSITIONAL_SPEED)
ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}globalRTHHeight').text = str(DEFAULT_GLOBAL_RTH_HEIGHT)

drone_info = ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}droneInfo')
ET.SubElement(drone_info, f'{{{WPML_NAMESPACE}}}droneEnumValue').text = '100'
ET.SubElement(drone_info, f'{{{WPML_NAMESPACE}}}droneSubEnumValue').text = '1'

ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}waylineAvoidLimitAreaMode').text = '0'

payload_info = ET.SubElement(mission_config, f'{{{WPML_NAMESPACE}}}payloadInfo')
ET.SubElement(payload_info, f'{{{WPML_NAMESPACE}}}payloadEnumValue').text = '99'
ET.SubElement(payload_info, f'{{{WPML_NAMESPACE}}}payloadSubEnumValue').text = '2'
ET.SubElement(payload_info, f'{{{WPML_NAMESPACE}}}payloadPositionIndex').text = '0'

# Create Folder
folder = ET.SubElement(document, 'Folder')

# Folder-level settings
ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}templateType').text = 'waypoint'
ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}templateId').text = '0'

wayline_param = ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}waylineCoordinateSysParam')
ET.SubElement(wayline_param, f'{{{WPML_NAMESPACE}}}coordinateMode').text = 'WGS84'
ET.SubElement(wayline_param, f'{{{WPML_NAMESPACE}}}heightMode').text = 'aboveGroundLevel'

ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}autoFlightSpeed').text = str(DEFAULT_SPEED)
ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}globalHeight').text = str(DEFAULT_GLOBAL_HEIGHT)
ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}caliFlightEnable').text = '0'
ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}gimbalPitchMode').text = 'manual'

global_heading = ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}globalWaypointHeadingParam')
ET.SubElement(global_heading, f'{{{WPML_NAMESPACE}}}waypointHeadingMode').text = 'followWayline'
ET.SubElement(global_heading, f'{{{WPML_NAMESPACE}}}waypointHeadingAngle').text = '0'
ET.SubElement(global_heading, f'{{{WPML_NAMESPACE}}}waypointPoiPoint').text = '0.000000,0.000000,0.000000'
ET.SubElement(global_heading, f'{{{WPML_NAMESPACE}}}waypointHeadingPathMode').text = 'followBadArc'
ET.SubElement(global_heading, f'{{{WPML_NAMESPACE}}}waypointHeadingPoiIndex').text = '0'

ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}globalWaypointTurnMode').text = 'toPointAndStopWithDiscontinuityCurvature'
ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}globalUseStraightLine').text = '1'

# Add Waypoints
for idx, row in df.iterrows():
    placemark = ET.SubElement(folder, 'Placemark')

    point = ET.SubElement(placemark, 'Point')
    ET.SubElement(point, 'coordinates').text = f"{row['longitude']},{row['latitude']}"

    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}index').text = str(idx)
    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}ellipsoidHeight').text = str(row['altitude'])
    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}height').text = str(DEFAULT_GLOBAL_HEIGHT)
    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}waypointSpeed').text = str(DEFAULT_SPEED)

    waypoint_heading = ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}waypointHeadingParam')
    ET.SubElement(waypoint_heading, f'{{{WPML_NAMESPACE}}}waypointHeadingMode').text = 'followWayline'
    ET.SubElement(waypoint_heading, f'{{{WPML_NAMESPACE}}}waypointHeadingAngle').text = '0'
    ET.SubElement(waypoint_heading, f'{{{WPML_NAMESPACE}}}waypointPoiPoint').text = '0.000000,0.000000,0.000000'
    ET.SubElement(waypoint_heading, f'{{{WPML_NAMESPACE}}}waypointHeadingPathMode').text = 'followBadArc'
    ET.SubElement(waypoint_heading, f'{{{WPML_NAMESPACE}}}waypointHeadingPoiIndex').text = '0'

    waypoint_turn = ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}waypointTurnParam')
    ET.SubElement(waypoint_turn, f'{{{WPML_NAMESPACE}}}waypointTurnMode').text = 'toPointAndStopWithDiscontinuityCurvature'
    ET.SubElement(waypoint_turn, f'{{{WPML_NAMESPACE}}}waypointTurnDampingDist').text = '0.2'

    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}useGlobalHeight').text = '1'
    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}useGlobalSpeed').text = '1'
    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}useGlobalHeadingParam').text = '1'
    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}useGlobalTurnParam').text = '1'
    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}useStraightLine').text = '1'

    # Action Group (minimal rotation/yaw)
    action_group = ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}actionGroup')
    ET.SubElement(action_group, f'{{{WPML_NAMESPACE}}}actionGroupId').text = str(idx)
    ET.SubElement(action_group, f'{{{WPML_NAMESPACE}}}actionGroupStartIndex').text = str(idx)
    ET.SubElement(action_group, f'{{{WPML_NAMESPACE}}}actionGroupEndIndex').text = str(idx)
    ET.SubElement(action_group, f'{{{WPML_NAMESPACE}}}actionGroupMode').text = 'sequence'

    action_trigger = ET.SubElement(action_group, f'{{{WPML_NAMESPACE}}}actionTrigger')
    ET.SubElement(action_trigger, f'{{{WPML_NAMESPACE}}}actionTriggerType').text = 'reachPoint'

    action = ET.SubElement(action_group, f'{{{WPML_NAMESPACE}}}action')
    ET.SubElement(action, f'{{{WPML_NAMESPACE}}}actionId').text = '0'
    ET.SubElement(action, f'{{{WPML_NAMESPACE}}}actionActuatorFunc').text = 'rotateYaw'
    action_param = ET.SubElement(action, f'{{{WPML_NAMESPACE}}}actionActuatorFuncParam')
    ET.SubElement(action_param, f'{{{WPML_NAMESPACE}}}aircraftHeading').text = '0'
    ET.SubElement(action_param, f'{{{WPML_NAMESPACE}}}aircraftPathMode').text = 'counterClockwise'

    ET.SubElement(placemark, f'{{{WPML_NAMESPACE}}}isRisky').text = '0'

# Add payload settings
payload_param = ET.SubElement(folder, f'{{{WPML_NAMESPACE}}}payloadParam')
ET.SubElement(payload_param, f'{{{WPML_NAMESPACE}}}payloadPositionIndex').text = '0'
ET.SubElement(payload_param, f'{{{WPML_NAMESPACE}}}focusMode').text = 'firstPoint'
ET.SubElement(payload_param, f'{{{WPML_NAMESPACE}}}meteringMode').text = 'average'
ET.SubElement(payload_param, f'{{{WPML_NAMESPACE}}}returnMode').text = 'singleReturnFirst'
ET.SubElement(payload_param, f'{{{WPML_NAMESPACE}}}samplingRate').text = '240000'
ET.SubElement(payload_param, f'{{{WPML_NAMESPACE}}}scanningMode').text = 'repetitive'
ET.SubElement(payload_param, f'{{{WPML_NAMESPACE}}}imageFormat').text = 'visable,ir'

# Save pretty KML
rough_string = ET.tostring(kml, 'utf-8')
reparsed = xml.dom.minidom.parseString(rough_string)
with open(OUTPUT_KML, 'w', encoding='utf-8') as f:
    f.write(reparsed.toprettyxml(indent="  "))

# Create KMZ
with zipfile.ZipFile(OUTPUT_KMZ, 'w', zipfile.ZIP_DEFLATED) as kmz:
    kmz.write(OUTPUT_KML)

# Cleanup (optional)
os.remove(OUTPUT_KML)

print(f"✅ Done! Created {OUTPUT_KMZ}")
