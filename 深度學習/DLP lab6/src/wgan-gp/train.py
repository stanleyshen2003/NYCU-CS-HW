import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import autograd
from PIL import Image

from tqdm import tqdm
from model import Discriminator, Generator, Classifier
from dataset import Iclevr_Dataset



class Agent():
    def __init__(self):
        self.epochs = 800
        lr_G = 5e-5
        betas_G = (0, 0.999)
        schedule_gamma = 0.3
        lr_D = 1e-4
        betas_D = (0, 0.999)
        self.batch_size = 64
        self.lambd = 10
        self.noise_dim = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = 'output_smaller_noise2'
        
        self.generator = Generator(noise_dim=self.noise_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.classifer = Classifier().to(self.device)
        
        self.start_epoch = 500
        self.pretrained_folder = 'output_smaller_nosie'
        if self.start_epoch > 0:
            self.generator.load_state_dict(torch.load(f'{self.pretrained_folder}/checkpoint/{self.start_epoch}/generator.pth'))
            self.discriminator.load_state_dict(torch.load(f'{self.pretrained_folder}/checkpoint/{self.start_epoch}/discriminator.pth'))
        
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion2 = torch.nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G, betas=betas_G)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D, betas=betas_D)
        
        self.scheduler_G = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_G, milestones=[200, 400], gamma=schedule_gamma)
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_D, milestones=[200, 400], gamma=schedule_gamma)
        
        self.writer = SummaryWriter(f'{self.output_dir}/logs')
        os.makedirs(self.output_dir+'/checkpoint', exist_ok=True)
    
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(self.batch_size, int(real_data.nelement()/self.batch_size)).contiguous()
        alpha = alpha.view(self.batch_size, 3, 64, 64)
        alpha = alpha.to(self.device)

        fake_data = fake_data.view(self.batch_size, 3, 64, 64)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(self.device)
        interpolates.requires_grad_(True)   

        disc_interpolates, _ = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)                              
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambd
        return gradient_penalty
    
    def train(self):
        train_dataset = Iclevr_Dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        
        
        for epoch in range(self.start_epoch, self.epochs+1):
            print('Epoch:', epoch)
            self.generator.train()
            self.discriminator.train()
            total_loss_D = 0
            total_loss_G = 0
            total_loss_C = 0

            with tqdm(total=len(train_dataset), ncols=150) as pbar:
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Train Discriminator
                    for p in self.discriminator.parameters():
                        p.requires_grad_(True)
                    for p in self.generator.parameters():
                        p.requires_grad_(False)
                    self.optimizer_D.zero_grad()
                    
                    # gen fake data and load real data
                    noise = torch.randn(images.size(0), self.noise_dim).to(self.device)
                    
                    fake_data = self.generator(noise, labels).detach()
                    images.requires_grad_(True)

                    # train with real data
                    disc_real, aux_output = self.discriminator(images)
                    classifier_loss = self.criterion(aux_output.view(-1), labels.view(-1)).mean()
                    disc_real = disc_real.mean()

                    # train with fake data
                    disc_fake, _ = self.discriminator(fake_data)
                    disc_fake = disc_fake.mean()

                    # train with interpolates data
                    gradient_penalty = self.calc_gradient_penalty(self.discriminator, images, fake_data)

                    # final disc cost
                    disc_cost = disc_fake - disc_real + gradient_penalty
                    disc_cost.backward(retain_graph=True)
                    classifier_loss.backward()
                    total_loss_D += disc_cost.item() + classifier_loss.item()
                    w_dist = disc_fake  - disc_real
                    self.optimizer_D.step()
                    
                    # Train Generator
                    if i % 5 == 0:
                        for p in self.discriminator.parameters():
                            p.requires_grad_(False)
                        for p in self.generator.parameters():
                            p.requires_grad_(True)

                        self.optimizer_G.zero_grad()
                        
                        # generate fake
                        noise = torch.randn(images.size(0), self.noise_dim).to(self.device)
                        noise.requires_grad_(True)
                        fake_data = self.generator(noise, labels)
                        gen_cost, gen_aux_output = self.discriminator(fake_data)
                        gen_cost = -gen_cost.mean()
                        
                        aux_errG = self.criterion(gen_aux_output.view(-1), labels.view(-1)).mean()
                        aux_errG = aux_errG * 10 * max(min(epoch / self.epochs,0.3), 1)
                        
                        g_cost = gen_cost + aux_errG
                        total_loss_C += aux_errG.item()
                        total_loss_G += gen_cost.item()
                        g_cost.backward()
                        self.optimizer_G.step()
                    pbar.set_postfix({ "wdist":w_dist.item(), "closs_D":classifier_loss.item(), "gloss":gen_cost.item(), "closs_G":aux_errG.item(), "class_loss2":class_loss2.item()}) 
                    pbar.update(images.size(0))
            
            
            self.writer.add_scalar('Loss/Discriminator', total_loss_D/len(train_loader), epoch)
            self.writer.add_scalar('Loss/Generator', total_loss_G/len(train_loader), epoch)
            self.writer.add_scalar('Loss/Classifier', total_loss_C/len(train_loader), epoch)
            if epoch % 5 == 0:
                # inference
                with torch.no_grad():
                    os.makedirs(f'{self.output_dir}/epoch_{epoch}', exist_ok=True)
                    for (images, labels) in train_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device).long()
                        
                        noise = torch.randn(images.size(0), self.noise_dim).to(self.device)
                        fake_images = self.generator(noise, labels)
                        fake_images = fake_images
                        fake_images = fake_images.detach().cpu()
                        # write the images to file
                        for i, img in enumerate(fake_images):
                            # i originally use transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                            inverse_transform = transforms.Compose([
                                transforms.Normalize((-1, -1, -1), (2, 2, 2)),
                                # transforms.Resize((240, 320)),
                                transforms.ToPILImage()
                            ])
                            img = inverse_transform(img)
                            img.resize((240, 320), Image.NEAREST)
                            img.save(f'{self.output_dir}/epoch_{epoch}/img_{i}.png')
                        break
                
            if epoch % 100 == 0:
                os.makedirs(f'{self.output_dir}/checkpoint/{epoch}', exist_ok=True)
                torch.save(self.generator.state_dict(), os.path.join(self.output_dir, 'checkpoint', str(epoch), 'generator.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(self.output_dir, 'checkpoint', str(epoch), 'discriminator.pth'))
                
                
if __name__ == '__main__':
    agent = Agent()
    agent.train()