# Peritoneal-Hemorrhage-Detection-and-Location
####本專題的資料集因為有隱私問題無法公開，如需資料集請聯絡國立中正大學資工系 熊博安 教授####

一、 CNN 實作部分：
在資料集的影像中，病人被分成rupt跟norm資料夾，分別代表病人腹腔CT有無出血。
資料夾中有一堆數字資料夾表示病人ID，該資料夾中會有他們腹部CT及標註的結合影像。
左邊部分是腹部CT影像，右邊部分是已經標註好出血位置的黑白影像。

先用Powershell以病人為單位隨機將rupt及norm資料夾中的病人資料夾打散，80%的放入train_set，其餘20%放入test_set及抓出他們所有影像。

model2的目標是去偵測有無出血。
根據目標資料夾位置去遍歷所有影像抓出來做resize，調成理想的大小。
再用切割將左右影像切開，右半部影像若有任何一個像素為白色，則程式中將之標記為1作為ground truth反之將之標記為0。
把所有training set中的影像送進去基礎的CNN模型中訓練，再進一步去測試及評估模型。

model3的目標是讓模型有能力去根據CT影像去標記出血位置。
1. 由於用這個模型會受到記憶體大小的限制所以在load image時加入batch size，讓其重組成一小包一小包的batch，每個epoch之間再讓他們去shuffle。
2. 對影像去前處理(調整大小及標準化)。
3. 將打包及處理好的影像餵進去U-Net訓練。
4. 最後將結果隨機抓出做視覺化。

二、 Mask RCNN實作部分:
本實作以Mask RCNN的model進行，以resnet50作為backbone network，進行出血偵測的模型訓練。

1. Dataset 資料集:

資料集分為norm(無出血)以及rupt(出血)，並且有CT原檔'image'資料夾及Label的'label'資料夾，我們將資料集混合後以「train 80%、test 20%」、在train中又分為「90% train + 10% valid」、test中norm以及rupt佔各半。


2. Label 資料標記:

觀察資料集中的label檔案，發現label的方式是以黑白圖片
在labelwithcv.py檔案中，我們用openCV的套件將label白色區塊的輪廓框出來，並轉換成labelme的標註形式，完成資料標記。

3. Model 模型:

我們使用Mask RCNN (resource: https://github.com/matterport/Mask_RCNN) 將資料送進model中使用Coco的pretrained weight做training，並跑了50 epoches。

在 PeritonealHemorrhageDetection資料夾中有train.py檔案，是用來設定mask RCNN的Custom Config及Custom Dataset，並呼叫上述Mask RCNN的function以進行訓練
