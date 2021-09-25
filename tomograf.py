import numpy as np
from skimage import exposure
from skimage.io import imread
from sklearn import preprocessing
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.transform import radon, iradon
import matplotlib.pylab as plt
import cv2
import math
from PIL import Image
import datetime
import ipywidgets as widgets
import pydicom
from pydicom.dataset import Dataset, FileDataset, validate_file_meta
from pydicom.uid import generate_uid
from pydicom._storage_sopclass_uids import MRImageStorage
from pydicom import dcmread
import os
from sklearn.preprocessing import MinMaxScaler

class Tomograf:
    def __init__(self, imgName, rozpietosc, iloscEmiterow, krok, process = None, iloscIteracji = None, withFilter = None):
        if process is None:
            self.process = False
        else:
            self.process = process
        if iloscIteracji is None:
            iloscIteracji = int(180/krok)
        if withFilter is None:
            self.withFilter = True
        else:
            self.withFilter = withFilter
        try:
            split_tup = os.path.splitext(imgName)
            if split_tup[1] =='.dcm':
            #dcm -> odczyt i zapis jako tablice
                ds = dcmread(imgName)
                keys = {x for x in dir(ds) if x[0].isupper()} - {'PixelData'}
                self.meta = {x: getattr(ds, x) for x in keys}
                self.img = image = ds.pixel_array
            elif split_tup[1]=='.jpg':
            #jpg -> konwertuje jako grayscale i zamienia w tablice
                self.xd = Image.open(imgName).convert('L')
                self.img = np.array(self.xd)

            self.w, self.h =self.img.shape
            self.krokAlfa=rozpietosc/iloscEmiterow
            self.krokSkanu=krok
            self.rozpietosc = rozpietosc
            self.iloscEmiterow = iloscEmiterow
            self.iloscSkanow = int(180/krok)
            xStart = self.w
            yStart =self.h
            self.srodekX=self.w/2
            self.srodekY=self.h/2
            self.filtr = False

            self.promien = self.w/2
            #self.promien = ((((self.srodekX - xStart )**2) + ((self.srodekY-yStart)**2) )**0.5)

            self.emiteryAll =[]
            self.detektoryAll=[]
            self.views = []
            self.filteredViews = []
            for x in range(self.iloscSkanow):
                self.views.append([])
                self.filteredViews.append([])

            self.reconstructed = np.zeros((self.w,self.h))
            self.kernel = []
            for i in range(-10,11,1):
                if(i==0):
                    self.kernel.append(1)
                elif(i%2==0):
                    self.kernel.append(0)
                else:
                    self.kernel.append(((-4/(math.pi**2))/(i**2)))        
            
        except AttributeError:
            print("Niepoprawna nazwa pliku")
        except FileNotFoundError:
            print("Niepoprawna nazwa pliku")
            
        if(self.process):
            self.wyznaczPozycjeEmiterow()
            self.imgReconstruction(self.withFilter,iloscIteracji)
            self.percentile()
            
    def wyznaczPozycjeEmiterow(self):
        nextScan = 0
        for k in range(self.iloscSkanow):
            #x = widgets.IntProgress(value=k,min=0,max=iloscSkanow,description='Loading:',style={'bar_color': 'maroon'},orientation='horizontal')
            #display(x)
            emitery=[]
            dekodery=[]
            sumAlfa = 0  
            rozpietoscPom = self.rozpietosc
            for i in range(self.iloscEmiterow):
                emitery.append([(math.cos(np.radians(sumAlfa+nextScan))*self.promien) + self.srodekX,(math.sin(np.radians(sumAlfa+nextScan))*self.promien) + self.srodekY])        
                dekodery.append([self.promien*math.cos(np.radians(180+rozpietoscPom+nextScan)) +self.srodekX, self.promien*math.sin(np.radians(180+rozpietoscPom+nextScan))+self.srodekY])
                rozpietoscPom = rozpietoscPom-self.krokAlfa
                sumAlfa+=self.krokAlfa
            count =0
            #jednen widok/skan z ilosciSkanow
            self.emiteryAll.append(emitery)
            self.detektoryAll.append(dekodery)
            for i,j in emitery:
                x1 = round(i)
                y1 = round(j)
                x2=round(dekodery[count][0])
                y2=round(dekodery[count][1])
                count+=1
                #tworzenie zwyklego sinogramu
                self.views[k].append(self.bresenhamAlgorithm(x1,y1,x2,y2))
            nextScan = nextScan + self.krokSkanu
                 
        for i in range(self.iloscSkanow):
            self.filteredViews[i] = np.convolve(self.views[i][:], self.kernel, mode='same')
            
    def bresenhamAlgorithm(self,x1, y1, x2, y2):
        sum = count =0
        kx = 1 if x1 < x2 else -1
        ky = 1 if y1 < y2 else -1

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if(0<=x1<=self.w and 0<=y1<=self.h):
            color = self.img[x1-1][y1-1]
            sum+=color
            count+=1
        if dx > dy:
            e = dx / 2
            for i in range(int(dx)):
                x1 += kx
                e -= dy
                if e < 0:
                    y1 += ky
                    e += dx
                if(0<=x1<=self.w and 0<=y1<=self.h):
                    color = self.img[x1-1][y1-1]
                    sum+=color
                    count+=1
        else:
            e = dy / 2
            for i in range(int(dy)):
                y1 += ky
                e -= dx
                if e < 0:
                    x1 += kx
                    e += dy
                if(0<=x1<=self.w and 0<=y1<=self.h):
                    color = self.img[x1-1][y1-1]
                    sum+=color
                    count+=1
        return (sum/count) if count!=0 else 0
    
    def reversedBrehensham(self,x1,y1,x2,y2,tab,raySum):
        kx = 1 if x1 < x2 else -1
        ky = 1 if y1 < y2 else -1

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if(0<=x1<=self.w and 0<=y1<=self.h):
            tab[x1-1,y1-1]+= raySum
        if dx > dy:
            e = dx / 2
            for i in range(int(dx)):
                x1 += kx
                e -= dy
                if e < 0:
                    y1 += ky
                    e += dx
                if(0<=x1<=self.w and 0<=y1<=self.h):
                    tab[x1-1,y1-1]+= raySum
        else:
            e = dy / 2
            for i in range(int(dy)):
                y1 += ky
                e -= dx
                if e < 0:
                    x1 += kx
                    e += dy
                if(0<=x1<=self.w and 0<=y1<=self.h):
                    tab[x1-1,y1-1]+= raySum
        return tab
    
    def imgReconstruction(self,withFilter, iteracje = None): 
        if iteracje is None:
            iteracje = self.iloscSkanow
        
        self.reconstructed = np.zeros((self.w,self.h))
        for k in range(iteracje):
            count = 0
            #print(f"{emiteryAll[k][count]}")
            for i,j in self.emiteryAll[k]:
                x1=round(i)
                y1=round(j)
                x2= round(self.detektoryAll[k][count][0])
                y2= round(self.detektoryAll[k][count][1])
                #print("emiter X1: "+str(x1)+" Y1: "+ str(y1) + " detektor X2: "+ str(x2)+" Y2: "+ str(y2))
                if withFilter:
                    self.reconstructed = self.reversedBrehensham(x1,y1,x2,y2, self.reconstructed, self.filteredViews[k][count])
                else:
                    self.reconstructed = self.reversedBrehensham(x1,y1,x2,y2, self.reconstructed ,self.views[k][count])
                count+=1
        if(self.process == False):
            plt.figure(figsize=(10,10))
            plt.imshow(self.reconstructed, cmap='gray'), plt.axis('off'),plt.title('reconstructed img', size=20)
        
    def wyswietlSinogram(self, zFiltrem):
        if zFiltrem:
            plt.figure(figsize=(10,10))
            plt.imshow(self.filteredViews, cmap='gray'), plt.axis('off'), plt.title('sinogram', size=20)
        else:
            plt.figure(figsize=(10,10))
            plt.imshow(self.views, cmap='gray'), plt.axis('off'), plt.title('sinogram', size=20)

    def processImage(self, imageName, step = None):
        if iteracje is None:
            iteracje = self.iloscSkanow
    
    def percentile(self):
        if(self.withFilter == True):
            lo, hi = np.percentile(self.reconstructed,(20,99))
        else:
            lo, hi = np.percentile(self.reconstructed,(1,99))
        self.reconstructed = exposure.rescale_intensity(self.reconstructed, in_range=(lo,hi))
        if(self.process == True):
            plt.figure()
            plt.imshow(self.reconstructed,cmap='gray')
            plt.show()        
def write_dicom(path, image, meta):    
    ds = Dataset()
    ds.MediaStorageSOPClassUID = MRImageStorage
    ds.MediaStorageSOPInstanceUID = generate_uid()
    ds.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    fd = FileDataset(path, {}, file_meta=ds, preamble=b'\0'*128)
    fd.is_little_endian = True
    fd.is_implicit_VR = False
    fd.SOPClassUID = MRImageStorage
    fd.PatientName = 'Test^Firstname'
    fd.PatientID = '123456'
    now = datetime.datetime.now()
    fd.StudyDate = now.strftime('%Y%m%d')
    fd.Modality = 'MR'
    fd.SeriesInstanceUID = generate_uid()
    fd.StudyInstanceUID = generate_uid()
    fd.FrameOfReferenceUID = generate_uid()
    fd.BitsStored = 16
    fd.BitsAllocated = 16
    fd.SamplesPerPixel = 1
    fd.HighBit = 15
    fd.ImagesInAcquisition = '1'
    fd.Rows = image.shape[0]
    fd.Columns = image.shape[1]
    fd.InstanceNumber = 1
    fd.ImagePositionPatient = r'0\0\1'
    fd.ImageOrientationPatient = r'1\0\0\0\-1\0'
    fd.ImageType = r'ORIGINAL\PRIMARY\AXIAL'
    fd.RescaleIntercept = '0'
    fd.RescaleSlope = '1'
    fd.PixelSpacing = r'1\1'
    fd.PhotometricInterpretation = 'MONOCHROME2'
    fd.PixelRepresentation = 1
    for key, value in meta.items():
        setattr(fd, key, value)
    validate_file_meta(fd.file_meta, enforce_standard=True)
    fd.PixelData = (image*255).astype(np.uint16).tobytes()
    fd.save_as(path, write_like_original=False) 