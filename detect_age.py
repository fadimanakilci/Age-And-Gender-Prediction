
import numpy as np
import argparse
import cv2
import os

# Bağımsız komut satırı argümanları belirlenir
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="programa girilen resim dizinine giden yol")
ap.add_argument("-f", "--face", required=True, help="yüz dedektörü model dizinine giden yol")
ap.add_argument("-a", "--age", required=True, help="yaş dedektörü model dizinine giden yol")
ap.add_argument("-g", "--gender", required=True, help="cinsiyet dedektörü model dizinine giden yol")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="zayıf tespitleri filtreleme için minimum olasılık")
args = vars(ap.parse_args())

# Dedektörün tahmin edeceği yaş sınıfları tanımlanır
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_BUCKETS = ["Kadin", "Erkek"]

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

# Resim yüklenir ve resim için bir input blobu oluşturulur
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Blob ağa verilir ve yüz tespit edilir
print("[BILGI] Yüz algılama...")
faceNet.setInput(blob)
detections = faceNet.forward()

# Bulununan yüz döngüye verilir
for i in range(0, detections.shape[2]):
	# tahminle ilişkili güven (yani olasılık) çıkarılır
	confidence = detections[0, 0, i, 2]

	# Güvenin(confidence) minimum güvenden daha yüksek olması sağlanarak zayıf algılamalar filtrelenir
	if confidence > args["confidence"]:
		# Nesne için sınırla kutunun (x, y) koordinatlarını hesaplayın
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Yüzün ROI'sini çıkarılır ve yüz ROI'sinden bir blob oluşturulur
		face = image[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
		global j
		j = i
		# Sınıflara ait tahminler bulunur ve en büyük olasılığa sahip yaş ve cinsiyet sınıfları belirlenir
		ageNet.setInput(faceBlob)
		predsA = ageNet.forward()
		i = predsA[0].argmax()
		age = AGE_BUCKETS[i]
		ageConfidence = predsA[0][i]
		genderNet.setInput(faceBlob)
		predsG = genderNet.forward()
		j = predsG[0].argmax()
		gender = GENDER_BUCKETS[j]
		genderConfidence = predsG[0][j]

		# Tahminler terminale yazdırılır
		text = "{}: {:.2f}%".format(age, ageConfidence * 100) + "  " + "{}: {:.2f}%".format(gender, genderConfidence * 100)
		print("[BILGI] {}".format(text))

		# İlgili sınıflar yazdırılır ve bulunan yüzün kutu sınırları çizilir
		y = startY - 20 if startY - 20 > 20 else startY + 20
		cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 0), 3)
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)

# Çıktı(output) resmi ekranda gösterilir
cv2.imshow("Image", image)
cv2.waitKey(0)