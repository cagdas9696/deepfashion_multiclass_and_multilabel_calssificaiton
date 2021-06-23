import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms,models
from torch import nn, optim
from torch.utils.data import Dataset
from test_deneme import accuracy
from model import MobileNetv2x, Densenet169, Resnet50
from losses import AsymmetricLoss


with open('/home/cyilmaz/deepfashion2/train.txt', 'r') as r:
    catl=r.readlines()

liste_cat = list()
print(len(catl))
for i in range(len(catl)):
    a = catl[i].split('/')[1]
    a = a.split('_')[-1]
    if a not in liste_cat:
        liste_cat.append(a)

liste_cat2 =['Tee', 'Tank', 'Dress', 'Shorts', 'Skirt', 'Blouse', 'Leggings', 'Hoodie', 'Sweater', 'Romper', 'Kimono', 'Jumpsuit', 'Jacket','Sweatpants','Jeans','Coat']

class dataset(Dataset):
    def __init__(self, img_txt, att_txt, transform=None):
        self.X = list()
        self.y_att = list()
        self.y_cat = list()

        with open(img_txt, 'r') as l:
            img_paths = l.readlines()
            
        with open(att_txt, 'r') as l:
            att_list = l.readlines()

        
        for line in range(len(img_paths)):
            img_path=img_paths[line].split('\n')[0]
           
            category_name=img_path.split('/')[1]
            category_name=category_name.split('_')[-1]
            if category_name not in liste_cat:
                continue 
            self.y_cat.append(liste_cat.index(category_name)) 
           
            #img_path=os.path.join('/home/cyilmaz/deepfashion2/', img_path)
            self.X.append(img_path)
            
            atts = att_list[line].split()
            labels = np.zeros(len(atts))
            for i in range(len(atts)):
                if int(atts[i]) == 1:
                    labels[i] = 1
            self.y_att.append(labels)    

        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        image = Image.open(self.X[idx])
        image = image.convert('RGB')
        label1 = np.array(self.y_att[idx]).astype('float')
        label2 = np.array(self.y_cat[idx]).astype('float')


        sample = {'image': image, \
                  'label_attribute': torch.from_numpy(label1), \
                  'label_category': torch.from_numpy(label2)
                  }
                    
        # Applying transformation
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

data_transforms = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.RandomRotation(degrees=(15)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

data_transforms2 = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


train_data = dataset('train.txt', 'train_attr.txt', transform=data_transforms)
test_data = dataset('val.txt', 'val_attr.txt', transform=data_transforms2)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=8, num_workers=2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model
model=Resnet50([45,26])
model.to(device)


# Loss Functions
#criterion1= AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(device)
criterion1= torch.nn.BCEWithLogitsLoss().to(device)

criterion2= nn.CrossEntropyLoss().to(device)
# Optimizer
optimizer = optim.SGD([{'params': model.parameters(), 'initial_lr': 0.01}], lr=0.01, momentum=0.7)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,15,40,50], gamma=0.1)



def train_model(model, criterion1,criterion2, optimizer, scheduler, n_epochs=50, max_epochs_stop=8):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    epochs_no_improve = 0

    for epoch in range(1, n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(train_dataloader):
            # importing data and moving to GPU
            image, label1, label2 = sample_batched['image'].to(device), \
                                    sample_batched['label_attribute'].to(device), \
                                    sample_batched['label_category'].to(device)

            image = torch.autograd.Variable(image)


            label1 = torch.autograd.Variable(label1)
            label1_ = label1.to(device)

            label2 = torch.autograd.Variable(label2)
            label2_ = label2.squeeze().type(torch.LongTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(image)
            label1_out = output['def'].to(device)
            label2_out = output['class'].to(device)

            # calculate loss
            loss1 = criterion1(label1_out, label1_)
            loss2 = criterion2(label2_out, label2_)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 4000 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                      (epoch, batch_idx + 1, train_loss))
        # validate the model #
        model.eval()
        for batch_idx, sample_batched in enumerate(test_dataloader):
            # importing data and moving to GPU
            image, label1, label2 = sample_batched['image'].to(device), \
                                    sample_batched['label_attribute'].to(device), \
                                    sample_batched['label_category'].to(device)
                    
            image = torch.autograd.Variable(image)

            label1 = torch.autograd.Variable(label1)
            label1_ = label1.to(device)
            #label2_ = label2.squeeze().type(torch.LongTensor).to(device)

            label2 = torch.autograd.Variable(label2)
            label2_ = label2.squeeze().type(torch.LongTensor).to(device)

            output = model(image)
            label1_out = output['def'].to(device)
            label2_out = output['class'].to(device)
            # calculate loss
            loss1 = criterion1(label1_out, label1_)
            loss2 = criterion2(label2_out, label2_)
            loss = loss1 + loss2


            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model, 'checkpoints/Resnet50_45_epoch_{}.pth'.format(epoch))

            ename = "checkpoints/Resnet50_45_epoch_{}.pth".format(epoch)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            valid_loss_min = valid_loss
            epochs_no_improve = 0
            best_epoch = epoch
            min_loss = valid_loss_min
        else:
            epochs_no_improve += 1
            if epochs_no_improve > max_epochs_stop:
                print("\nEarly stopping! total epochs:{}. Best epoch:{}. Best Loss:{} ".format(epoch, best_epoch,
                                                                                               min_loss))
                break

        scheduler.step()
    return model,ename




if __name__=='__main__':
    _,best_model=train_model(model, criterion1, criterion2, optimizer, scheduler, n_epochs=50, max_epochs_stop=5)

    model_path = "{}".format(best_model)
    imgs_txt = 'val.txt'
    labels_txt = 'val_attr.txt'
    l1,l2,l3=accuracy(model_path, imgs_txt, labels_txt, 256)
    

