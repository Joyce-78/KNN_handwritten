
import cv2
import numpy as np

# doc anh
digits = cv2.imread("C:/Users/Admin/Code/Python/digits.png", cv2.IMREAD_GRAYSCALE)
test_digits = cv2.imread("C:/Users/Admin/Code/Python/test_digits.png", cv2.IMREAD_GRAYSCALE)
width = 20
height = 20
dim = (width, height)

# resize image
#inImage = cv2.resize(digits, dim, interpolation= cv2.INTER_AREA)
inImage = digits
# doc anh training 
test_digits = cv2.imread("C:/Users/Admin/Code/Python/test_digits.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', inImage)
cv2.waitKey(0)
#cat anh thanh nhieu anh nho
cells = [np.hsplit(row, 100) for row in np.vsplit(test_digits,50)]

#chuyen dinh dang anh thanh dinh dang mang
trainingimage = np.array(cells)
predictionimage = np.array(inImage)

#chuan hoa mang , chuyen thanh mang 1 chieu
train = trainingimage[:, :100].reshape(-1, 400).astype(np.float32)
test = predictionimage.reshape(-1, 400).astype(np.float32)

#tao mang 
k = np.arange(10)

#lap lai tung phan tu cua mang nay 500 lan
train_labels = np.repeat(k,500)[:, np.newaxis]

#khoi tao Knn
knn = cv2.ml.KNearest_create()

#dua anh va ket qua vao
knn.train(train, 0, train_labels)
result = knn.findNearest(test, 5)
print (result[0])
print(result[2][0])