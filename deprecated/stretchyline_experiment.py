from imports import *
from models import *
from data import *
from utils import *


class StretchyLine_Experiment():
    def __init__(self, experiment_name, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.experiment_name = experiment_name
        self.config = deepcopy(config)
        
        output_dir = './stretchyline_checkpoints/' + self.experiment_name
        os.makedirs(output_dir, exist_ok=True)
        
        self.checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')
        self.config_path = os.path.join(output_dir, 'config.txt')
        
        self.history = []
        self.train_loss = []
        self.test_loss = []
        
        self.model = self.config['model_class']().to(self.device)
        self.optimizer = self.config['optimizer_class'](self.model.parameters(), lr=self.config['lr'])
        self.loss_fn = self.config['loss_fn']
        
        self.config['model_class'] = self.model
        self.config['optimizer_class'] = self.optimizer
        
        dataset_args = deepcopy(self.config['dataset_args'])
        dataset_args.update(train=True)
        
        self.train_loader = DataLoader(
            self.config['dataset_class'](**dataset_args),
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True,
        )
        
        dataset_args.update(train=False)
        
        self.test_loader = DataLoader(
            self.config['dataset_class'](**dataset_args),
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True,
        )
        
        # load checkpoint and check compatibility 
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "Checkpoint found with same name but different config."
                    )
                    
            self.load()
        else:
            self.save()
            
    @property
    def epoch(self):
        return len(self.history)
    
    def setting(self):
        return self.config
    
    def __repr__(self):
        string = ''
        
        for key, val in self.setting().items():
            string += '{}: {}\n'.format(key, val)
            
        return string
    
    def state_dict(self):
        return {
            'model' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'history' : self.history,
            'train loss' : self.train_loss,
            'test loss' : self.test_loss,
        }
    
    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.history = checkpoint['history']
        self.train_loss = checkpoint['train loss']
        self.test_loss = checkpoint['test loss']
        
    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)
        
        with open(self.config_path, 'w') as f:
            print(self, file=f)
            
    def load(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.load_state_dict(checkpoint)
        
        del checkpoint
        
    def plot(self, clear=False):
        if clear:
            display.display(plt.clf())
            display.clear_output(wait=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        
        axes = axes.flatten()
        
        axes[0].clear()
        axes[1].clear()
        
        axes[0].plot(self.train_loss)
        axes[0].set_title('Training loss', size=16, color='teal')
        axes[0].set_xlabel('Epochs', size=16, color='teal')
        axes[0].grid()
        
        axes[1].plot(self.test_loss)
        axes[1].set_title('Testing loss', size=16, color='teal')
        axes[1].set_xlabel('Epochs', size=16, color='teal')
        axes[1].grid()
        
        plt.show()
        
    def train_epoch(self):
        self.model.train()
        
        losses = []
        
        for img1, img2, _ in self.train_loader:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            recons = self.model(img1, img2)

            if isinstance(self.loss_fn, SSIM_Regular_Loss):
                loss = self.loss_fn(recons.unsqueeze(1), img2)
            elif isinstance(self.loss_fn, SSIM_Contrastive_Loss):
                loss = self.loss_fn(recons.unsqueeze(1), img1, img2)
            else:
                loss = self.loss_fn(recons.unsqueeze(1), img2)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.detach().cpu().item())
        
        return np.mean(losses)
    
    def test_epoch(self):
        self.model.eval()
        
        losses = []
        
        with torch.no_grad():
            for img1, img2, _ in self.test_loader:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)        

                recons = self.model(img1, img2)

                if isinstance(self.loss_fn, SSIM_Regular_Loss):
                    loss = self.loss_fn(recons.unsqueeze(1), img2)
                elif isinstance(self.loss_fn, SSIM_Contrastive_Loss):
                    loss = self.loss_fn(recons.unsqueeze(1), img1, img2)
                else:
                    loss = self.loss_fn(recons.unsqueeze(1), img2)
                
                losses.append(loss.detach().cpu().item())
        
        return np.mean(losses)
    
    def train(self, num_epochs, show_plot):
        if show_plot:
            self.plot(clear=False)
        
        print(self)
        print('Start/Continue training from epoch {0}'.format(self.epoch))
        
        while self.epoch < num_epochs:            
            self.train_loss.append(self.train_epoch())
            
            if (self.epoch % 5) == 0:                
                self.test_loss.append(self.test_epoch())
            
            self.history.append(self.epoch + 1)
            
            if show_plot:
                self.plot(clear=True)
            else:
                print("Epoch: {0}".format(self.epoch))
                print("Train Loss: {:.4f}".format(self.train_loss[-1]))
                print("Test Loss: {:.4f}".format(self.test_loss[-1]))
                print()
            
            thread = Thread(target=self.save)
            thread.start()
            thread.join()
            
        print('Finished training\n')
