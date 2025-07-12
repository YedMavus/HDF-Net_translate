import numpy as np
import torch
from torchvision.transforms import transforms as T
from loss import total_loss
from net import HDF_Net
from torch import optim, nn
from dataset import LiverDataset
from torch.utils.data import DataLoader

# Whether to use current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BatchSize=4
epoches=100

x_transform = T.Compose([
    T.Resize((256,256)),  # Resize to 256x256
    # T.Scale((256,256)),
    # T.RandomHorizontalFlip(),  # Random horizontal flip
    # T.RandomVerticalFlip(),    # Random vertical flip
    T.ToTensor(),  # Convert to Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])
# mask only needs to be converted to tensor
y_transform = T.Compose([
    T.Resize((256,256)),
    # T.RandomHorizontalFlip(),  # Random horizontal flip
    # T.RandomVerticalFlip(),    # Random vertical flip
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# definition of accuracy function
def accuracy(output:torch.Tensor , mask):
    output=torch.sigmoid(output)
    output = (output > 0.5).float()
    error = torch.sum(torch.abs(output - mask))
    acc = 1 - error / (BatchSize * mask.shape[2] * mask.shape[3])
    return acc

if __name__ == '__main__':

    train_dataset=LiverDataset(r"F:\ImageManipulationDatasets\train-NIST\train", transform=x_transform, target_transform=y_transform)
    trainloader=DataLoader(train_dataset,batch_size=BatchSize,num_workers=8,shuffle=True)
    test_dataset=LiverDataset(r"F:\ImageManipulationDatasets\test-NIST\test", transform=x_transform, target_transform=y_transform)
    testloader=DataLoader(test_dataset,num_workers=8)
    # print the length of training set
    print('len(trainloader):{}'.format(len(trainloader)))
    # print the length of test set
    print('len(testloader):{}'.format(len(testloader)))

    model=HDF_Net().to(device)
    # model.load_state_dict(torch.load("./pre_rgb_path/weights_47.pth", map_location='cpu'))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
    num_steps = len(trainloader) * epoches
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-7, last_epoch=-1)

    trainACC=[]
    testACC=[]
    globalLoss=[]
    bestACC=0.0
    best_epoch=0
    for epoch in range(epoches):
        print("===Starting this Epoch {}===Total Epochs {}===".format(epoch,epoches-1))
        # Collect training parameters
        epochAccuracy = []
        epochLoss = []
        model.train()
        # =============Actual training process=================
        for batch_id, (img, mask) in enumerate(trainloader):
            # torch.train()
            # First initialize gradient to 0
            optimizer.zero_grad()
            img=img.to(device)
            mask=mask.to(device)
            # edge=edge.to(device)
            output1,output2,output3 = model(img)
            loss = total_loss(output1, mask)+total_loss(output2,mask)/4+total_loss(output3,mask)/8
            # loss=total_loss(output1,label)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(lr_scheduler.last_epoch + 1)
            epochAccuracy.append(accuracy(output1, mask).cpu())
            epochLoss.append(loss.item())  # need to get the numerical value for conversion
            if batch_id % (int(len(trainloader) / 20)) == 0:
                print("Currently running [{}/{}], Current Epoch accuracy: {:.6f}%, Loss: {:.6f}".format(batch_id, len(trainloader),
                                                                                 np.mean(epochAccuracy) * 100, loss))
        # ==============End of current training round==============
        # Collect training set accuracy
        trainACC.append(np.mean(epochAccuracy))
        globalLoss.append(np.mean(epochLoss))
        # ==========Perform one test set evaluation============
        localTestACC = []
        model.eval()  # Enter evaluation mode, save resources
        for img, mask in testloader:
            torch.no_grad()  # Context manager, no gradient tracking in this section
            img=img.to(device)
            mask=mask.to(device)
            # edge=edge.to(device)
            output,_,_ = model(img)
            localTestACC.append(accuracy(output, mask).cpu())
        # # ==========End of test set evaluation================
        # # Collect test set accuracy
        testACC.append(np.mean(localTestACC))
        print("Current Epoch finished, training accuracy: {:.6f}%, training loss: {:.6f}, test accuracy: {:.6f}%".format(trainACC[-1] * 100,globalLoss[-1], testACC[-1] * 100))
        # print("Current Epoch finished, training accuracy: {:.6f}%, training loss: {:.6f}".format(trainACC[-1] * 100, globalLoss[-1]))
        # Periodically save results to file
        if bestACC < testACC[-1]:
            bestACC=testACC[-1]
            best_epoch = epoch + 1
        # if epoch == epoches - 1 or (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), "./nist_train/weights_%d.pth" % (best_epoch))

        if epoch == 99:
            torch.save(model.state_dict(), "./nist_train/weights_%d.pth" % (epoch+1))





