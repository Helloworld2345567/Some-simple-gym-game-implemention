# Some-simple-gym-game-implemention
**My Envirnment：**\
Python3.10\
torch 2.0\
numpy 1.24.0\
gym 0.26.2\
gym  atari and box2d dependency\
_Don't worry about envirnment problems.High compatibility it has._

**install gym:**
```
pip install gym
pip install gym[atari]
pip install gym[accept-rom-license]
apt-get install -y swig
pip install box2d-py
pip install gym[box2d] 
```

## Pong , Lunerland , AirRaid 
These game's action is discreate, use simple DQN algorithms to implement it.



## BipedalWalker 
This game's action is continuous, so you can use PPO or SAC to implement it.DQN is not good at it
<img src="https://github.com/Helloworld2345567/Some-simple-gym-game-implemention/blob/master/BipedalWalkerHardcore-SAC/gif/result_hard6000.gif" alt="animation">

### SAC：
[**Site Reference**](https://github.com/CoderAT13/BipedalWalkerHardcore-SAC.git)
![Result](https://github.com/Helloworld2345567/Some-simple-gym-game-implemention/blob/master/BipedalWalkerHardcore-SAC/imgs/SAC-4.jpg)

```python
import​ ​os
import​ ​csv
import​ ​torch
import​ ​numpy​ ​as​ ​np
from​ ​torch.utils.data​ ​import​ ​Dataset,​ ​DataLoader
from​ ​sklearn.model_selection​ ​import​ ​train_test_split


class​ ​SignalDataset(Dataset):
​ ​​ ​​ ​​ ​def​ ​__init__(self,​ ​idx,​ ​data,​ ​transform=None):
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​self.idx​ ​=​ ​idx
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​self.data​ ​=​ ​data
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​self.transform​ ​=​ ​transform

​ ​​ ​​ ​​ ​def​ ​__len__(self):
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​return​ ​len(self.idx)

​ ​​ ​​ ​​ ​def​ ​__getitem__(self,​ ​index):
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​sample_idx​ ​=​ ​self.idx[index]
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​sample​ ​=​ ​self.data[sample_idx]
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​label​ ​=​ ​sample[-1]
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​sample​ ​=​ ​sample[:-1]

​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​if​ ​self.transform:
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​sample​ ​=​ ​self.transform(sample)

​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​return​ ​torch.tensor(sample).float(),​ ​label

def​ ​load_data(csv_path,​ ​test_size=0.1,​ ​random_state=42):
​ ​​ ​​ ​​ ​file_data​ ​=​ ​[]
​ ​​ ​​ ​​ ​for​ ​file_path​ ​in​ ​os.listdir(csv_path):
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​with​ ​open(os.path.join(csv_path,​ ​file_path),​ ​'r')​ ​as​ ​f:
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​reader​ ​=​ ​csv.reader(f)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​for​ ​row​ ​in​ ​reader:
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​file_data.append(np.array(row,​ ​dtype=np.float32))

​ ​​ ​​ ​​ ​data_np​ ​=​ ​np.vstack(file_data)
​ ​​ ​​ ​​ ​X_train,​ ​X_test,​ ​y_train,​ ​y_test​ ​=​ ​train_test_split(data_np[:,​ ​:-1],​ ​data_np[:,​ ​-1],​ ​test_size=test_size,​ ​random_state=random_state)

​ ​​ ​​ ​​ ​train_dataset​ ​=​ ​SignalDataset(np.arange(len(X_train)),​ ​X_train)
​ ​​ ​​ ​​ ​test_dataset​ ​=​ ​SignalDataset(np.arange(len(X_test)),​ ​X_test)

​ ​​ ​​ ​​ ​return​ ​train_dataset,​ ​test_dataset

def​ ​signal_augmentation(batch_size,​ ​data):
​ ​​ ​​ ​​ ​augmented_data​ ​=​ ​np.stack(np.split(data,​ ​2,​ ​axis=1),​ ​axis=-1)
​ ​​ ​​ ​​ ​
​ ​​ ​​ ​​ ​#​ ​确保返回的数据是维度​ ​(batch_size,​ ​2048,​ ​2)
​ ​​ ​​ ​​ ​return​ ​torch.tensor(augmented_data).float()
```

2.​ ​定义​ ​ResNet18​ ​架构和自监督对比学习训练方法：

```python
import​ ​torch.nn​ ​as​ ​nn
import​ ​torch.optim​ ​as​ ​optim
from​ ​torchvision.models​ ​import​ ​resnet18


def​ ​get_resnet_model():
​ ​​ ​​ ​​ ​resnet​ ​=​ ​resnet18(pretrained=False)
​ ​​ ​​ ​​ ​resnet.conv1​ ​=​ ​nn.Conv2d(2,​ ​64,​ ​kernel_size=7,​ ​stride=2,​ ​padding=3,​ ​bias=False)
​ ​​ ​​ ​​ ​resnet.fc​ ​=​ ​nn.Sequential(nn.Linear(512,​ ​512),​ ​nn.ReLU(),​ ​nn.Linear(512,​ ​6))
​ ​​ ​​ ​​ ​
​ ​​ ​​ ​​ ​return​ ​resnet.cuda()

def​ ​contrastive_loss(features):
​ ​​ ​​ ​​ ​#​ ​自监督对比学习损失函数
​ ​​ ​​ ​​ ​features​ ​=​ ​nn.functional.normalize(features,​ ​dim=1)

​ ​​ ​​ ​​ ​sim_matrix​ ​=​ ​torch.matmul(features,​ ​features.T)
​ ​​ ​​ ​​ ​sim_matrix.fill_diagonal_(0)
​ ​​ ​​ ​​ ​
​ ​​ ​​ ​​ ​max_sim_per_row​ ​=​ ​torch.max(sim_matrix,​ ​dim=1)[0].detach()
​ ​​ ​​ ​​ ​exp_sim_matrix​ ​=​ ​torch.exp(sim_matrix​ ​-​ ​max_sim_per_row[:,​ ​None])
​ ​​ ​​ ​​ ​
​ ​​ ​​ ​​ ​tar_rowsum_one​ ​=​ ​torch.ones((features.size(0),​ ​1))​ ​-​ ​torch.eye(features.size(0))
​ ​​ ​​ ​​ ​row_sum_reverse​ ​=​ ​exp_sim_matrix.matmul(tar_rowsum_one).sum(dim=1)

​ ​​ ​​ ​​ ​loss​ ​=​ ​-torch.log(exp_sim_matrix.diag()​ ​/​ ​row_sum_reverse).mean()

​ ​​ ​​ ​​ ​return​ ​loss
```

3.​ ​设置训练和验证循环：

```python
def​ ​train_model(model,​ ​train_loader,​ ​test_loader,​ ​epochs=100,​ ​learning_rate=0.001):
​ ​​ ​​ ​​ ​optimizer​ ​=​ ​optim.Adam(model.parameters(),​ ​lr=learning_rate)

​ ​​ ​​ ​​ ​best_acc​ ​=​ ​0
​ ​​ ​​ ​​ ​best_model​ ​=​ ​None
​ ​​ ​​ ​​ ​train_loss_history​ ​=​ ​[]
​ ​​ ​​ ​​ ​val_accuracy_history​ ​=​ ​[]

​ ​​ ​​ ​​ ​for​ ​epoch​ ​in​ ​range(epochs):
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​model.train()
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​train_loss​ ​=​ ​0.0
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​for​ ​i,​ ​(inputs,​ ​_)​ ​in​ ​enumerate(train_loader):
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​inputs​ ​=​ ​signal_augmentation(batch_size,​ ​inputs)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​inputs​ ​=​ ​inputs.cuda()

​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​optimizer.zero_grad()
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​features,​ ​_​ ​=​ ​model(inputs)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​loss​ ​=​ ​contrastive_loss(features)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​loss.backward()
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​optimizer.step()

​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​train_loss​ ​+=​ ​loss.item()
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​if​ ​(i​ ​+​ ​1)​ ​%​ ​10​ ​==​ ​0:
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​print("周%d\tEpoch​ ​[%d/%d],​ ​Loss:​ ​%.4f"​ ​%​ ​(i​ ​+​ ​1,​ ​epoch​ ​+​ ​1,​ ​epochs,​ ​train_loss​ ​/​ ​(i​ ​+​ ​1)))

​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​train_loss_history.append(train_loss​ ​/​ ​len(train_loader))
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​val_accuracy​ ​=​ ​evaluate(model,​ ​test_loader)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​val_accuracy_history.append(val_accuracy)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​if​ ​val_accuracy​ ​>​ ​best_acc:
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​best_acc​ ​=​ ​val_accuracy
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​best_model​ ​=​ ​model

​ ​​ ​​ ​​ ​return​ ​best_model,​ ​train_loss_history,​ ​val_accuracy_history

def​ ​evaluate(model,​ ​test_loader):
​ ​​ ​​ ​​ ​correct​ ​=​ ​0
​ ​​ ​​ ​​ ​total​ ​=​ ​0
​ ​​ ​​ ​​ ​model.eval()
​ ​​ ​​ ​​ ​with​ ​torch.no_grad():
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​for​ ​inputs,​ ​labels​ ​in​ ​test_loader:
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​inputs​ ​=​ ​signal_augmentation(batch_size,​ ​inputs)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​inputs​ ​=​ ​inputs.cuda()
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​labels​ ​=​ ​labels.cuda()

​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​_,​ ​outputs​ ​=​ ​model(inputs)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​_,​ ​predicted​ ​=​ ​torch.max(outputs.data,​ ​1)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​total​ ​+=​ ​labels.size(0)
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​correct​ ​+=​ ​(predicted​ ​==​ ​labels).sum().item()

​ ​​ ​​ ​​ ​return​ ​correct​ ​/​ ​total
```

4.​ ​把一切都放在一起：

```python
import​ ​matplotlib.pyplot​ ​as​ ​plt

if​ ​__name__​ ​==​ ​'__main__':
​ ​​ ​​ ​​ ​csv_path​ ​=​ ​'./path/to/csv/files'
​ ​​ ​​ ​​ ​batch_size​ ​=​ ​32

​ ​​ ​​ ​​ ​train_dataset,​ ​test_dataset​ ​=​ ​load_data(csv_path)
​ ​​ ​​ ​​ ​train_loader​ ​=​ ​DataLoader(train_dataset,​ ​batch_size=batch_size,​ ​shuffle=True,​ ​num_workers=4)
​ ​​ ​​ ​​ ​test_loader​ ​=​ ​DataLoader(test_dataset,​ ​batch_size=batch_size,​ ​shuffle=False,​ ​num_workers=4)

​ ​​ ​​ ​​ ​resnet_model​ ​=​ ​get_resnet_model()
​ ​​ ​​ ​​ ​best_model,​ ​train_loss_history,​ ​val_accuracy_history​ ​=​ ​train_model(resnet_model,​ ​train_loader,​ ​test_loader)

​ ​​ ​​ ​​ ​plt.figure()
​ ​​ ​​ ​​ ​plt.plot(range(1,​ ​len(train_loss_history)​ ​+​ ​1),​ ​train_loss_history,​ ​label='Training​ ​Loss')
​ ​​ ​​ ​​ ​plt.xlabel('Epoch')
​ ​​ ​​ ​​ ​plt.ylabel('Loss')
​ ​​ ​​ ​​ ​plt.legend()
​ ​​ ​​ ​​ ​plt.show()

​ ​​ ​​ ​​ ​plt.figure()
​ ​​ ​​ ​​ ​plt.plot(range(1,​ ​len(val_accuracy_history)​ ​+​ ​1),​ ​val_accuracy_history,​ ​label='Validation​ ​Accuracy')
​ ​​ ​​ ​​ ​plt.xlabel('Epoch')
​ ​​ ​​ ​​ ​plt.ylabel('Accuracy')
​ ​​ ​​ ​​ ​plt.legend()
​ ​​ ​​ ​​ ​plt.show()
```
