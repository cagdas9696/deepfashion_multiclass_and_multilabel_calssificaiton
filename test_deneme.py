import os
import sys
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import time


def testClassifier(model_path="checkpoints/Densenet169_16_epoch_5.pth", img_path="img/Island_Graphic_Boxy_Tee/img_00000002.jpg", img_size=256):



    data_transforms = transforms.Compose([transforms.Resize((img_size,img_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("\nloading multi-output classification...")
    model = torch.load(model_path)
    model.to(device)


    t = time.time()
    
    image = Image.open(img_path)
    image = image.convert('RGB')
    image = data_transforms(image)

    image = image.to(device)
    image = torch.autograd.Variable(image.unsqueeze(0))
    t = time.time()

    output = model(image)
    
    elapsed = time.time()
    ftime = float(elapsed - t)

    dinamik1x_out = output['class'].to(device)
    dinamik1_out = torch.argmax(dinamik1x_out, dim=1).unsqueeze(1)[0][0]
    top3 = torch.topk(dinamik1x_out.unsqueeze(1)[0][0], 3, dim=0)[1].tolist()


    dinamik2_out = output['def'].to(device)
    dinamik2x_out = torch.sigmoid(dinamik2_out)[0]

    dinamik2_out = torch.round(dinamik2x_out).to(torch.int)
    
    cat_list=['Tee', 'Tank', 'Dress', 'Shorts', 'Skirt', 'Blouse','Leggings', 'Hoodie','Sweater', 'Romper','Kimono', 'Jumpsuit','Jacket', 'Sweatpants', 'Jeans', 'Coat']
    att_list=["floral","graphic","striped","embroidered","pleated","solid","lattice","long_sleeve","short_sleeve","sleeveless","maxi_length","mini_length","no_dress","crew_neckline","v_neckline","square_neckline","no_neckline","denim","chiffon","cotton","leather","faux","knit","tight","loose","conventional"]
    
    dinamik1_class =cat_list[dinamik1_out]
    conf2=list()
    top3_list = list()
    dinamik1x_out = dinamik1x_out.tolist()
    for i in top3:
        conf2.append(dinamik1x_out[0][i])
        top3_list.append(cat_list[i])
        
    conf2=torch.sigmoid(torch.tensor([conf2]))[0].tolist()
    conf2=[ round(elem*100,2) for elem in conf2]
    
    
    dinamik2x_out = dinamik2x_out.tolist()
    dinamik2x_out = [ round(elem,2) for elem in dinamik2x_out]
    özellik=list()
    conf=list()
    for i in range(len(dinamik2_out)):
        if dinamik2_out[i] ==1:
            conf.append(dinamik2x_out[i]*100)
            özellik.append(att_list[i])


    print("\nPrediction_category: %{} {}\n".format(conf2[0],dinamik1_class))
    print("Prediction_attribute : {} \nPrediction_atribute_confidence: {}\n".format(özellik,conf))
    print("Prediction_top3_category : {} \nPrediction_top3_category_confidence: {}\n".format(top3_list,conf2))

    return [dinamik2_out,ftime]


def accuracy(model_path="MobileNetv3_large.pth",imgs_txt="val.txt",labels_txt="labels.txt",img_size=256):

    data_transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("\nloading multi-output classification...")
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    with open(imgs_txt,'r') as f:
        imgs_path=f.readlines()
    
    with open('deepfashion2/train.txt','r') as f:
        imgs_path2=f.readlines()

    cat_list =list()    
    for i in imgs_path2:
        a = i.split('/')[1]
        a = a.split('_')[-1]
        if a not in cat_list:
            cat_list.append(a)
    
    cat_list2=['Tee','Tank', 'Dress', 'Shorts', 'Skirt', 'Blouse','Leggings', 'Hoodie','Sweater', 'Romper','Kimono', 'Jumpsuit','Jacket', 'Sweatpants', 'Jeans', 'Coat']
    
    #cat_list.sort()
    print(cat_list)    
    with open(labels_txt,'r') as f:
        att_list = f.readlines()
        
    acc_att=0
    pre=0
    rec=0
    tp = 0
    true_cat = 0
    false_cat = 0
    t = time.time()
    for line in range(len(imgs_path)):

        #img_pathx = imgs_path[line].split('\n')[0]
        #img_path = os.path.join('/home/cyilmaz/deepfashion2',img_pathx)
        
        img_path = imgs_path[line].split('\n')[0]

        if img_path.split('/')[1] =='Striped_A-line_Dress':
            continue
        
        category = img_pathx.split('/')[1]
        category = category.split('_')[-1]
        if category not in cat_list:
            continue

        image = Image.open(img_path)
        image = image.convert('RGB')
        image = data_transforms(image)

        image = image.to(device)
        image = torch.autograd.Variable(image.unsqueeze(0))

        output = model(image)
        
        dinamik1_out = output['class'].to(device)
        dinamik1_out = torch.argmax(dinamik1_out, dim=1).unsqueeze(1)[0][0]


        dinamik2_out = output['def'].to(device)
        dinamik2_out = torch.sigmoid(dinamik2_out)[0]
        dinamik2_out = torch.round(dinamik2_out).to(torch.int)

        
        if cat_list.index(category) == dinamik1_out:
            true_cat += 1
    
        else:
            false_cat += 1

        
        atts = att_list[line].split()
        actual = np.zeros(len(atts))
        
        for i in range(len(atts)):
                if int(atts[i]) == 1:
                    actual[i] = 1
        
        actual=torch.from_numpy(actual).to(device)
        

        true_att = torch.sum(dinamik2_out == actual)

        for i in range(len(actual)):
            if actual[i] == 1 and actual[i] == dinamik2_out[i]:
                tp += 1


        acc_att += true_att.item()/26
        rec += torch.sum(actual).item()
        pre += torch.sum(dinamik2_out).item()

    elapsed = time.time()
    ftime = float(elapsed - t)

    cat_accuracy = (true_cat /(true_cat + false_cat)) * 100
    rec_accuracy = (tp / rec) * 100
    pre_accuracy = (tp / pre) * 100
    #att_accuracy = (acc_att / 3310) * 100 for 16 category 
    att_accuracy = (acc_att / len(att_list)) * 100
    print('speed: {}'.format(ftime))
    print('category acc: {}'.format(cat_accuracy))
    print('attribute acc: {} attribute_recall: {}  attribute_precision : {}'.format(att_accuracy,rec_accuracy,
                                                                                    pre_accuracy))
    return [att_accuracy,rec_accuracy,pre_accuracy]

if __name__=='__main__':
    imgs_txt='test.txt'
    att_txt= 'test_attr.txt'
    model_path = "checkpoints/Densenet169_16_epoch_5.pth"
    img_path= "img/Island_Graphic_Boxy_Tee/img_00000002.jpg"
    #l1,l2,l3 =accuracy(model_path, imgs_txt, att_txt, 256)
    l1,l2 = testClassifier(model_path, img_path, 256)
