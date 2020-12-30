
from imutils.video import VideoStream
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import imutils
import time
import cv2
import os

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def detect_and_predict_age(frame, faceNet, ageNet, genderNet, minConf=0.5):
	# dedektörün tahmin edeceği yaş sınıflarının listesi tanımlanır
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
	# 0=male 1=female
	GENDER_BUCKETS = ["Kadin", "Erkek"]
	results = []
	# shape(yükseklik × genişlik × diğer boyutlar) [:2] ise ilk iki öğeyi alır(h, w)
	(h, w) = frame.shape[:2]
	# blob, ortalama çıkarma, normalleştirme ve kanal değiştirme işleminden sonra oluşan girdi resmi
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	# Yüz tespiti elde etmek için ağa blob girdi olarak verilir
	faceNet.setInput(blob)
	# forward() ile katman çıktısını hesaplamak için katmanlarda ilerlenir
	detections = faceNet.forward()

	# Tahmin edilen yüz çerçevesinin içerisinde dolaşılır
	for i in range(0, detections.shape[2]):
		# Tespit için bir güven(olasılık) değeri oluşturulur
		confidence = detections[0, 0, i, 2]
		# Güvenin minumum güvenden fazla olmasını sağlayarak zayıf tespitler filtrelenir
		if confidence > minConf:
			# Nesne için sınırlama kutusunun (x, y) koordinatları hesaplanır
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# Yüzün yatırım getirisi(ROI) çıkarılır
			face = frame[startY:endY, startX:endX]
			# Çerçevedeki yanlış yüz algılamalarını filtrelemek için yüz YG'si yeterince büyük değerle karşılaştırılır
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue
			faceBlob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
			global j
			j = i
			# Yaş hakkında tahminler yapılır ve en büyük olasılığa sahip olan sınıf atanır
			ageNet.setInput(faceBlob)
			predsA = ageNet.forward()
			i = predsA[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = predsA[0][i]
			# Cinsiyet hakkında tahminler yapılır ve en büyük olasılığa sahip olan sınıf atanır
			genderNet.setInput(faceBlob)
			predsG = genderNet.forward()
			j = predsG[0].argmax()
			gender = GENDER_BUCKETS[j]
			genderConfidence = predsG[0][j]
			# yüz kutu sınırları ve tahmin edilen sınıflar tutulur
			d = {"loc": (startX, startY, endX, endY), "age": (age, ageConfidence), "gender": (gender, genderConfidence)}
			# Sonuç listesi güncellenir
			results.append(d)
	return results


# Bağımsız komut satırı argümanları belirlenir
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="yüz dedektörü model dizinine giden yol")
ap.add_argument("-a", "--age", required=True, help="yaş dedektörü model dizinine giden yol")
ap.add_argument("-g", "--gender", required=True, help="cinsiyet dedektörü model dizinine giden yol")
ap.add_argument("-c", "--conf", type=float, default=0.5, help="confider zayıf tespitleri filtreleme için minimum olasılık")
args = vars(ap.parse_args())

# Modeller yüklenir
print("[BILGI] Yüz dedektör modeli yükleniyor...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[BILGI] Yaş dedektör modeli yükleniyor...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[BILGI] Cinsiyet dedektör modeli yükleniyor...")
prototxtPath = os.path.sep.join([args["gender"], "gender.prototxt"])
weightsPath = os.path.sep.join([args["gender"], "gender_net.caffemodel"])
genderNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# Video akışı başlatılır
print("[BILGI] Video akışı başlatılıyor...")
vs = VideoStream(src=0).start()
# Kamera sensörünün ısınması için zaman verilir
time.sleep(3.0)

while True:
	# Çerçeveli video akışını yakala
	frame = vs.read()
	# 300*300 olan çerçeve boyutunu yeniden boyutlandırır
	frame = imutils.resize(frame, width=800)
	# çerçevedeki yüzleri algıla ve her yüz için yaşı tahmin et
	results = detect_and_predict_age(frame, faceNet, ageNet, genderNet, minConf=args["conf"])
	for r in results:
		# ilişkili yaşla birlikte yüzün sınırlayıcı kutusunu çizilir
		text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100) + "  " + "{}: {:.2f}%".format(r["gender"][0], r["gender"][1] * 100)
		(startX, startY, endX, endY) = r["loc"]
		y = startY - 20 if startY - 20 > 20 else startY + 20
		cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 3)
		cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (205, 205, 0), 2)
	# frame çıktısını ekranda göster
	cv2.imshow("Frame", frame)
	# Klavyeden basılan tuşun ASCII değeri key değişkenine atanmış olur
	key = cv2.waitKey(1) & 0xFF
	# close anlamına gelen "c" tuşuna basılırsa frame kapatılır
	if key == ord("c"):
		break

# windowstaki cv2 değişkenine ait her şeyi temizler
cv2.destroyAllWindows()
vs.stop()
