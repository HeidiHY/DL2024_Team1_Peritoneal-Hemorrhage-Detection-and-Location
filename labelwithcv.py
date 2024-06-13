import cv2
import json
import os
import base64
import subprocess
'''
資料處理：label
使用OpenCV物件外框辨識將label做標記，並且輸出為labelme格式的資料

'''
def labeling_with_cv2(path):

    filename = os.path.splitext(os.path.basename(path))[0]
    # print(filename)

    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, bin = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    contour , _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes":[],
        "imagePath": os.path.basename(path),
        "imageData": None,
        "imageHeight": image.shape[0],
        "imageWidth": image.shape[1],
    }
    for c in contour:
        eps = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        points = approx.reshape(-1,2).tolist()

        labelme_data["shapes"].append({
            "label" : "rupt",                     
            "points": points,
            "group_id": 'null',
            "description": "",
            "shape_type": "polygon",
            "flags": {}                    
        })
    with open(path, 'rb') as img:
        img_byte = img.read()
        img_base64 = base64.b64encode(img_byte).decode('utf-8')

    labelme_data['imageData'] = img_base64

    output_dir = os.path.join(os.path.abspath(os.path.join(path, '../..')), 'json')
    # print(output_dir + '\\' + filename + '.json')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_dir + '\\' + filename + '.json', 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)

def transfer(path):
    path += '/json/'
    jsons = os.listdir(path)
    for j in jsons:
        dir = path + j
        subprocess.run(['labelme_json_to_dataset' , dir])
        break
    
# base path
path = 'C:/Users/wt090/Desktop/DL_final/training_dataset/'
t , v = path+os.listdir(path)[0], path+os.listdir(path)[1]

directories = os.listdir(t) # 全部的資料夾名
for i in range(len(directories)):
    # directories[i] = t +'/'+ directories[i] + '/label/'
    directories[i] = t +'/'+ directories[i] 


valid_dir = os.listdir(v)
for i in range(len(valid_dir)):
    # directories.append(v +'/'+ valid_dir[i] + '/label/') 
    directories.append(v +'/'+ valid_dir[i]) 
    

for dir in directories:
    imgs = os.listdir(dir)
    print(dir)
    for img in imgs:
        # labeling_with_cv2(dir + '/' + img)
        transfer(dir)
        break
    break
    

print('done')