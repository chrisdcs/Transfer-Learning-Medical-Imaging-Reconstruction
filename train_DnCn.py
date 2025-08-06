from utils.UMRI_model import *


mode = "domain"
if mode == "anatomy":
    anatomy = 'prostate'
    n = 40
elif mode == "sampling":
    anatomy = '10'
    n = 100
elif mode == "dataset":
    anatomy = 'stanford'
    n = 100
elif mode == "domain":
    anatomy = 'cifar10'
    n = 400

model = DnCn().cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.95)


acc = 5
mask = 'cartesian'



if mode == "anatomy":
    #dataset = universal_data(['data/brain/brain_singlecoil_train.mat', 'data/knee/knee_singlecoil_train.mat'], 
                            # 'data/cardiac/cardiac_singlecoil_train.mat'], 
    #                        acc=acc, mask=mask)
    dataset = anatomy_data(f'data/{anatomy}/{anatomy}_singlecoil_train.mat', acc=acc, mask = mask, n=n)
elif mode == "sampling":
    #anatomy = 'brain'
    file = f'data/{anatomy}/brain_singlecoil_train.mat'
    # file = f'data/{anatomy}/knee_singlecoil_train.mat' # could be changed to other things like knee etc.
    dataset = universal_sampling_data(file, [10, 5, 3.33], "cartesian")
elif mode == "dataset":
    # this is for cross-dataset transfer learning
    #anatomy = 'imagenet'
    file = f'data/{anatomy}/{anatomy}_singlecoil_train.mat' # could be changed to other things like knee etc.
    dataset = anatomy_data(file, acc=acc, mask = "cartesian", n=n)
elif mode == "domain":
    file = f'data/{anatomy}/{anatomy}_singlecoil_train.mat' # could be changed to other things like knee etc.
    dataset = anatomy_data(file, acc=acc, mask=mask, n=n)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

if mode == "anatomy":
    save_dir = f"universal_MRI/{anatomy}/cross_anatomy/checkpoints_{acc}_{mask}_samples_{n}"
elif mode == "sampling":
    save_dir = f"universal_MRI/{anatomy}/cross_sampling/checkpoints_{anatomy}_{mask}_samples_{n}"
elif mode == "dataset":
    save_dir = f"universal_MRI/{anatomy}/cross_dataset/checkpoints_{anatomy}_{mask}_samples_{n}"
elif mode == "domain":
    save_dir = f"universal_MRI/{anatomy}/cross_domain/checkpoints_{anatomy}_{mask}_samples_{n}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


n_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for epoch in range(1, n_epochs+1):
    PSNR_list = []
    loss_list = []
    
    for i, data in enumerate(loader):
        
        im_und, k_und, mask, img_gnd, k_gnd = data
        
        im_und = im_und.to(device)
        k_und = k_und.to(device)
        mask = mask.to(device)
        img_gnd = img_gnd.to(device)
        k_gnd = k_gnd.to(device)    
        
        optim.zero_grad()
        
        output = model(im_und, k_und, mask)
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
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_%d.pth' % (epoch)))
        print('Model saved\n')