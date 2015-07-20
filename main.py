import sys
try:
	from PIL import Image
	import numpy as np
	import pylab as plt
except :
	print ("Falló importar: PIL, numpy, pylab")
	sys.exit(0)
from config import *


def open_image(name, default_path = True):
	if default_path:
		tmp  = np.array(Image.open(TRAINING_PATH+name))
	else : tmp = np.array(Image.open(name))
	w, h = len(tmp), len(tmp[0])
	img  = np.zeros(w*h)
	l = 0
	for i in range(w):
		for j in range(h):
			img[l] = tmp[i][j][0] 
			l += 1
	return img

def save_img(array_, name,  width=50, height=50 ):
	tmp = array_.reshape((width, height))
	img_array = np.ones((width, height, 3), dtype=np.uint8)
	for i in range(width):
		for j in range(height):
			img_array[i][j] *= tmp[i][j]
	img = Image.fromarray(img_array)
	img.save(OUTPUT_PATH+name)

def train_model(n):
	X = []
	for name in range(1, n+1):
		X.append(open_image(str(name)+IMG_FORMAT))
	return np.array(X)

def main(detect_name):
	detect = open_image(detect_name, False)
	print("Entrenando modelo...")
	X = train_model(TRAINING_ELEMENTS)
	print("Se ha creado una matriz de %dx%d."%(len(X), len(X[0])))
	M = np.zeros(len(X[0]))
	for element in X:
		M += element
	M /= TRAINING_ELEMENTS
	detect -= M
	S = []
	for x in X:
		S.append(x-M)
	S = np.array(S)
	U, t, V = np.linalg.svd(S, full_matrices=False)
	W = np.dot(V, detect - M)
	i = 1
	flag = 0
	print("Comparando con:")
	for s in S:
		tmp = np.dot(V, s - M)
		norma = np.linalg.norm(tmp - W)/1000
		print("Imagen: "+ str(i) +".jpg \t", norma)
		if norma < 1.:
			flag = i
		i += 1
	if flag :
		print("\n Se encontró una coinsidencia: "+str(flag)+".jpg")
		Image.open(TRAINING_PATH+str(flag)+".jpg").show()
		Image.open(detect_name).show()
	else : print("No se encontraron coincidencias.")

if __name__ == '__main__':
	if len(sys.argv)!=2:
		print("Uso: python main.py <imagen_buscada>.jpg")
		sys.exit(0)
	main(sys.argv[1])