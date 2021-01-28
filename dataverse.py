import pandas as pd
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/home/petros/Documents/Dataverse/wheather_data.csv')
data
data.head()

data.info()
data.describe().T

sns.countplot(data.location_id)
sns.countplot(data.source_id)

data.groupby('value_type_id').nunique()
data.groupby('value').nunique()
data['value'].corr(data['value_type_id'])

#data1 = data[data.value_type_id == 11]
#data2 = data[data.value_type_id == 12]
#data1.describe().T
#data2.describe().T

#sns.distplot(data1.value)
#sns.distplot(data2.value)

# Categorical variables

categorical = data.select_dtypes(include = ['object']).keys()
print(categorical)

quantitive = data.select_dtypes(include = ['int64', 'float64']).keys()
print(quantitive)

#Transform Timestamp

data['Date'] = pd.to_datetime(data['timestamp'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['hour'] = data['Date'].dt.hour
data.head()
data.info()

df = data.drop(['id','timestamp','Date'], axis=1)

df['identifier'].nunique()
df['location_id'].nunique()


df['identifier'] = df['identifier'].astype('category')
df['identifier'] = df['identifier'].cat.codes

df.head()



X = df.iloc[:,1:]
y =  df.iloc[:,0]

# Train - Test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=21)


scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

def get_class_distribution(obj):
    count_dict = {
        "rating_3": 0,
        "rating_4": 0,
        "rating_5": 0,
        "rating_6": 0,
        "rating_7": 0,
        "rating_8": 0,
    }

    for i in obj:
        if i == 3:
            count_dict['rating_3'] += 1
        elif i == 4:
            count_dict['rating_4'] += 1
        elif i == 5:
            count_dict['rating_5'] += 1
        elif i == 6:
            count_dict['rating_6'] += 1
        elif i == 7:
            count_dict['rating_7'] += 1
        elif i == 8:
            count_dict['rating_8'] += 1
        else:
            print("Check classes.")

    return count_dict

pd.DataFrame.from_dict([get_class_distribution(y_train)])



y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)

class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)


train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
num_features = len(X.columns)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=num_features)
test_loader = DataLoader(dataset=test_dataset, batch_size=num_features)

num_features
class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()

        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, num_features)

        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)

def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)




#Check for GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


model = MultipleRegression(num_features)
model.to(device)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_stats = {
    'train': [],
    'val': []
}


for i in tqdm(range(1, EPOCHS + 1)):

    #Training
    train_epoch_loss = 0

    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()


    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0

        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

            val_epoch_loss += val_loss.item()

    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))


    print(f'Epoch {i+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')

train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
plt.figure(figsize=(15,8))
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
