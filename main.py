import torch #TensorFlowなどと同じ機械学習ライブラリ
import cv2 #OpenCV 画像加工、機械学習
import numpy as np #演算ライブラリ
from mss import mss #スクリーンショットライブラリ
import time #時間計測ライブラリ

model = torch.hub.load('ultralytics/yolov5', 'yolov5s',force_reload=False) 

#テキスト色の設定
red=0,0,255
green=0,255,0
blue=255,0,0

with mss() as sct:
    monitor = {"top": 0, "left": 0, "width": 600, "height":1000}

    while True:
        start_time = time.perf_counter()

        screenshot = np.array(sct.grab(monitor)) 
        results = model(screenshot, size=600)

        #物体の数だけカウント
        person_count=0
        #推論結果を取得
        obj = results.pandas().xyxy[0]
        frame=results.ims[0]

        #バウンディングボックスの情報を取得
        for i in range(len(obj)):
            name = obj.name[i]
            if name == "person":
                xmin = int(obj.xmin[i])
                ymin = int(obj.ymin[i])
                xmax = int(obj.xmax[i])
                ymax = int(obj.ymax[i])

                person_count+=1
                cv2.putText(frame, str(name), (xmin, ymin-15), cv2.FONT_HERSHEY_TRIPLEX, 1, (red), 2)
                cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(red),3,)

        cv2.putText(frame, f"persons {person_count}", (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (green), 2)
        cv2.putText(frame, f"FPS: {int(1/(time.perf_counter() - start_time))}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (green), 2)

        #opencvでウィンドウ作成
        cv2.imshow("frame", frame)
        
        #qキーで全て終了
        if(cv2.waitKey(1) == ord('q')):
            cv2.destroyAllWindows()
            break