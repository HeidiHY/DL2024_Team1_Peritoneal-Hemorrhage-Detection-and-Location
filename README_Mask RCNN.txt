Mask RCNN實作部分:

本實作以Mask RCNN的model進行，以resnet50作為backbone network，進行出血偵測的模型訓練。

1. Dataset 資料集:

資料集分為norm(無出血)以及rupt(出血)，並且有CT原檔'image'資料夾及Label的'label'資料夾，我們將資料集混合後以「train 80%、test 20%」、在train中又分為「90% train + 10% valid」、test中norm以及rupt佔各半。


2. Label 資料標記:

觀察資料集中的label檔案，發現label的方式是以黑白圖片
在labelwithcv.py檔案中，我們用openCV的套件將label白色區塊的輪廓框出來，並轉換成labelme的標註形式，完成資料標記。

3. Model 模型:

我們使用Mask RCNN(resource: https://github.com/matterport/Mask_RCNN)將資料送進model中使用Coco的pretrained weight做training，並跑了50 epoches。


