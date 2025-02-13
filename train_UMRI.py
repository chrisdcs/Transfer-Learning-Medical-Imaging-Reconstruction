from utils.UMRI_model import *

anatomies = ['brain', 'knee']
model = UDnCn(anatomies).cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.95)


acc = 5
mask = 'radial'
dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 'data/knee/knee_singlecoil_train.mat'], 
                         acc=acc, mask=mask)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

folder = os.path.join("universal_MRI", "universal", f"{mask}_{acc}")

if not os.path.exists(folder):
    os.makedirs(folder)


n_epochs = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for epoch in range(1, n_epochs+1):
    PSNR_list = []
    loss_list = []
    
    for i, data in enumerate(loader):
        
        im_und, k_und, mask, img_gnd, k_gnd, anatomy = data
        
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)    
        
        optim.zero_grad()
        output = model(im_und, k_und, mask, anatomy[0])
        output = torch.abs(output).clamp(0, 1)
        
        img_gnd = torch.abs(img_gnd)
        
        loss = torch.sum(torch.square(output - img_gnd))
        loss.backward()
        optim.step()
        
        loss_list.append(loss.item())
        
        for j in range(output.shape[0]):
            PSNR_list.append(psnr(output[j].cpu().detach().numpy(), img_gnd[j].cpu().detach().numpy()))
        if (i+1) % 100 == 0:
            print(i+1, loss.item())
    scheduler.step()
    avg_l = np.mean(loss_list)
    avg_p = np.mean(PSNR_list)
    epoch_data = ' [Epoch %02d/%02d] Avg Loss: %.4f \t Avg PSNR: %.4f\n\n' % \
            (epoch, n_epochs, avg_l, avg_p)
    print(epoch_data)
    
    if (epoch) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(folder, 'model_%d.pth' % (epoch)))
        print('Model saved\n')