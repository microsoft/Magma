import re


# is instruction English
def is_english_simple(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        click_point = floats
    elif len(floats) == 4:
        click_point = [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    return click_point

# bbox (qwen str) -> bbox
def extract_bbox(s):
    # Regular expression to find the content inside <box> and </box>
    pattern = r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>"
    matches = re.findall(pattern, s)
    # Convert the tuples of strings into tuples of integers
    return [(int(x.split(',')[0]), int(x.split(',')[1])) for x in sum(matches, ())]

def extract_mark_id(s):
    match = re.search(r'Mark: (\d+)', s)
    if match:
        return int(match.group(1))
    return None    