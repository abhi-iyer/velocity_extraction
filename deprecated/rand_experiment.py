from imports import *
from models import *
from data import *
from utils import *


class Rand_Experiment():
    def __init__(self, experiment_name, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.experiment_name = experiment_name
        self.config = deepcopy(config)
        
        output_dir = './rand_checkpoints/' + self.experiment_name
        os.makedirs(output_dir, exist_ok=True)
        
        self.checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')
        self.config_path = os.path.join(output_dir, 'config.txt')
        
        self.history = []
        self.train_loss = []
        self.train_subloss = defaultdict(list)
        self.test_loss = []
        self.test_subloss = defaultdict(list)
        
        self.model = self.config['model_class'](**self.config['model_args']).to(self.device)
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
            'train subloss' : self.train_subloss,
            'test loss' : self.test_loss,
            'test subloss' : self.test_subloss,
        }
    
    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.history = checkpoint['history']
        self.train_loss = checkpoint['train loss']
        self.train_subloss = checkpoint['train subloss']
        self.test_loss = checkpoint['test loss']
        self.test_subloss = checkpoint['test subloss']

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
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 12), constrained_layout=True)
        
        axes = axes.flatten()
        
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[3].clear()
        
        axes[0].plot(self.train_loss)
        axes[0].set_title('Training loss', size=16, color='teal')
        axes[0].set_xlabel('Epochs', size=16, color='teal')
        axes[0].grid()
        
        axes[1].plot(self.test_loss)
        axes[1].set_title('Testing loss', size=16, color='teal')
        axes[1].set_xlabel('Epochs', size=16, color='teal')
        axes[1].grid()

        if len(self.train_subloss) == 0:
            axes[2].set_visible(False)
        else:
            for (key, value) in self.train_subloss.items():
                axes[2].plot(value, label=key)
            axes[2].set_title('Training subloss', size=16, color='teal')
            axes[2].set_xlabel('Epochs')
            axes[2].legend()
            axes[2].grid()

        if len(self.test_subloss) == 0:
            axes[3].set_visible(False)
        else:
            for (key, value) in self.test_subloss.items():
                axes[3].plot(value, label=key)
            axes[3].set_title('Testing subloss', size=16, color='teal')
            axes[3].set_xlabel('Epochs')
            axes[3].legend()
            axes[3].grid()
        
        plt.show()
        
    def train_epoch(self):
        self.model.train()
        
        losses = []
        sublosses = defaultdict(list)
        
        for i1, i2, i3, _ in self.train_loader:
            i1 = i1.to(self.device)
            i2 = i2.to(self.device)
            i3 = i3.to(self.device)

            loss = None
            loss_categories = None

            if isinstance(self.loss_fn, ContrastV):
                #v_32, pred_i1_1 = self.model(i3.clone().detach().squeeze(), i2.clone().detach().squeeze())
                #v_11, pred_i1_2 = self.model(i1.clone().detach().squeeze(), i1.clone().detach().squeeze())

                #v_22, pred_i2 = self.model(i2.clone().detach().squeeze(), i2.clone().detach().squeeze())
                
                v_12, pred_i3_1 = self.model(i1.clone().detach().squeeze(), i2.clone().detach().squeeze())
                #v_33, pred_i3_2 = self.model(i3.clone().detach().squeeze(), i3.clone().detach().squeeze())

                loss, loss_categories = self.loss_fn(
                    None, None, # predicted i1
                    None, # predicted i2
                    pred_i3_1.unsqueeze(1), None, # predicted i3
                    i1, i2, i3, # ground-truth images
                    None, None, None, # attraction terms
                    v_12, None, # repel terms
                )

                # loss, loss_categories = self.loss_fn(
                #     pred_i1_1.unsqueeze(1), pred_i1_2.unsqueeze(1), # predicted i1
                #     pred_i2.unsqueeze(1), # predicted i2
                #     pred_i3_1.unsqueeze(1), pred_i3_2.unsqueeze(1), # predicted i3
                #     i1, i2, i3, # ground-truth images
                #     v_11, v_22, v_33, # attraction terms
                #     v_12, v_32, # repel terms
                # )
            else:
                _, pred_i3 = self.model(i1.squeeze(), i2.squeeze())

                loss = self.loss_fn(pred_i3.unsqueeze(1), i3)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach().cpu().item())
            if loss_categories and len(loss_categories):
                for (key, value) in loss_categories.items():
                    sublosses[key].append(value.detach().cpu().item())

        for (key, value) in sublosses.items():
            sublosses[key] = np.mean(value)
        
        return np.mean(losses), sublosses
    
    def test_epoch(self):
        self.model.eval()
        
        losses = []
        sublosses = defaultdict(list)
        
        with torch.no_grad():
            for i1, i2, i3, _ in self.test_loader:
                i1 = i1.to(self.device)
                i2 = i2.to(self.device)
                i3 = i3.to(self.device)

                loss = None
                loss_categories = None

                if isinstance(self.loss_fn, ContrastV):
                    #v_32, pred_i1_1 = self.model(i3.clone().detach().squeeze(), i2.clone().detach().squeeze())
                    #v_11, pred_i1_2 = self.model(i1.clone().detach().squeeze(), i1.clone().detach().squeeze())

                    #v_22, pred_i2 = self.model(i2.clone().detach().squeeze(), i2.clone().detach().squeeze())
                    
                    v_12, pred_i3_1 = self.model(i1.clone().detach().squeeze(), i2.clone().detach().squeeze())
                    #v_33, pred_i3_2 = self.model(i3.clone().detach().squeeze(), i3.clone().detach().squeeze())

                    loss, loss_categories = self.loss_fn(
                        None, None, # predicted i1
                        None, # predicted i2
                        pred_i3_1.unsqueeze(1), None, # predicted i3
                        i1, i2, i3, # ground-truth images
                        None, None, None, # attraction terms
                        v_12, None, # repel terms
                    )

                    # loss, loss_categories = self.loss_fn(
                    #     pred_i1_1.unsqueeze(1), pred_i1_2.unsqueeze(1), # predicted i1
                    #     pred_i2.unsqueeze(1), # predicted i2
                    #     pred_i3_1.unsqueeze(1), pred_i3_2.unsqueeze(1), # predicted i3
                    #     i1, i2, i3, # ground-truth images
                    #     v_11, 22, v_33, # attraction terms
                    #     v_12, v_32, # repel terms
                    # )
                else:
                    _, pred_i3 = self.model(i1.squeeze(), i2.squeeze())

                    loss = self.loss_fn(pred_i3.unsqueeze(1), i3)
                
                losses.append(loss.detach().cpu().item())
                if loss_categories and len(loss_categories):
                    for (key, value) in loss_categories.items():
                        sublosses[key].append(value.detach().cpu().item())
        
        for (key, value) in sublosses.items():
            sublosses[key] = np.mean(value)
        
        return np.mean(losses), sublosses
    
    def train(self, num_epochs, show_plot):
        if show_plot:
            self.plot(clear=False)
        
        print(self)
        print('Start/Continue training from epoch {0}'.format(self.epoch))
        
        while self.epoch < num_epochs:
            loss, subloss = self.train_epoch()
            self.train_loss.append(loss)
            for (key, value) in subloss.items():
                self.train_subloss[key].append(value)
            
            if (self.epoch % 5) == 0:
                loss, subloss = self.test_epoch()
                self.test_loss.append(loss)
                for (key, value) in subloss.items():
                    self.test_subloss[key].append(value)
            
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
