import os
import csv
from PIL import Image

fireTrainData = './dataClone/Data/Train_Data/Fire/'
nonFireTrainData = './dataClone/Data/Train_Data/Non_Fire/'

fireTestData = './dataClone/Data/Test_Data/Fire/'
nonFireTestData = './dataClone/Data/Test_Data/Non_Fire/'

newTrainCsv = 'fireTrainData.csv'
newTestCsv = 'fireTestData.csv'

fireTrainImages = os.listdir(fireTrainData)
nonFireTrainImages = os.listdir(nonFireTrainData)

fireTestImages = os.listdir(fireTestData)
nonFireTestImages = os.listdir(nonFireTestData)

def make_files():
    with open(newTrainCsv, mode="w", newline='') as file:
        writer = csv.writer(file)
        for fireImage in fireTrainImages:
            writer.writerow([fireImage, "1"])
        for nonFireImage in nonFireTrainImages:
            writer.writerow([nonFireImage, "0"])

    with open(newTestCsv, mode="w", newline='') as file:
        writer = csv.writer(file)
        for fireImage in fireTestImages:
            writer.writerow([fireImage, "1"])
        for nonFireImage in nonFireTestImages:
            writer.writerow([nonFireImage, "0"])

def removeCorruptedFile(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError):
        print(f"Image corrupted")
        os.remove(file_path)
        return True
    
def removeAllCorrupted(dir):
    corrupted = 0
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            result = removeCorruptedFile(file_path)
            if result:
                corrupted += 1
    print(f"Removed {str(corrupted)} corrupted files")

#removeAllCorrupted(fireTrainData)
#removeAllCorrupted(nonFireTrainData)
#removeAllCorrupted(fireTestData)
#removeAllCorrupted(nonFireTestData)

make_files()