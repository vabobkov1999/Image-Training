import ast
import mahotas
import cv2
def get_names(path = 'features/name.txt'):
    with open(path, 'r') as filehandle:
        names = filehandle.read()
    names = ast.literal_eval(names) # парсим список
    return names

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # перекрашиваем рисунок
    feature = cv2.HuMoments(cv2.moments(image)).flatten() # получаем фичу с делаем одномерным списком
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0) # получаем фичу и берем среднее по строкам
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()