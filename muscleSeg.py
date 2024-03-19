import sys
import json
from torchvision import transforms as T
import torch
import pydicom
import numpy as np
import PIL.Image as Image
import cv2

def cnt_perimeter(cnt):
        perimeter = cv2.arcLength(cnt,True)
        return perimeter

class MuscleSeg(object):
    def __init__(self, model_path, device):
            self.model=torch.load(model_path).eval().to(device)
            self.Transform= T.Compose([T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.classname=['background','rectus abdominus muscle','trans abd,int and ext obl','psoas major mucle',
           'quardratus lumborum muscle','eretor spinae muscle','L3 Vertebral body']
            
    def seg(self, dicom_file):
            ds = pydicom.read_file(dicom_file,force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            image = ds.pixel_array

            image_h, image_w, = image.shape
            intercept = ds.RescaleIntercept
            slope = ds.RescaleSlope
            img = slope*image+intercept

            img[img<-190]=-190
            img[img>150]=150

            if np.nanmin(img) < 0:
                img += abs(np.nanmin(img))+1
            else:
                img -= abs(np.nanmin(img))+1

            img *= 255/np.nanmax(img)
            img = Image.fromarray(img)
            img = img.convert('RGB')

            #predict
            input = self.Transform(img).unsqueeze(0).to(device)
            pred = self.model(input).squeeze(0).detach().cpu()
            pred = np.argmax(pred, axis=0)
            shapes = []
            for i in range(1, 7):
                mask = pred[pred==i]
                layer = np.array(pred==i, np.uint8)*255 
                contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours=list(contours)
                contours.sort(key = cnt_perimeter, reverse=True)
                try:
                    contour = contours[0].reshape(-1, 2).tolist()
                except:
                    contour = []

                shape_data = {'label': self.classname[i],
                              'points': contour,
                              'shape_type': 'polygon',
                             }
                shapes.append(shape_data)
                for ctlen in range(1,len(contours)):
                    ctsh1 = contours[ctlen].shape
                    if ctsh1[0]>10:
                        contour = contours[ctlen].reshape(-1, 2).tolist()
                        shape_data = {'label': self.classname[i],
                                    'points': contour,
                                    'shape_type': 'polygon',
                                    }
                        shapes.append(shape_data)
            inference_response={'imagePath':dicom_file,'shapes':shapes}
            
            with open(dicom_file[:-4]+'.json', "w") as jsonf:
                json.dump(inference_response, jsonf)
            jsonf.close



if __name__ == "__main__":
    
    device=sys.argv[1]
    dcm=sys.argv[2]
    segModel=MuscleSeg('model.pth',device)
    segModel.seg(dcm)
    