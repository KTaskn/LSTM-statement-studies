import numpy as np
import torch
import torch.nn as nn
import string
from tqdm import tqdm
import torch.optim as optim

def gen(num):
    N = 200
    for i in range(num):
        s = np.random.randint(0, len(text) - N)
        yield (
            torch.tensor(text[s:s+N]).unsqueeze(0), 
            torch.tensor(text[s+1:s+N+1]).unsqueeze(0)
        )

N_HIDDEN = 100
N_EMB = 100
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_CHAR, N_EMB)
        self.rnn = nn.LSTM(N_EMB, N_HIDDEN, 2, batch_first=True)
#         self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer = nn.Linear(N_HIDDEN, N_CHAR)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.rnn(x)
        x = self.layer(x)
#         return self.softmax(x)
        return x

if __name__ == "__main__":
    all_characters = string.printable
    N_CHAR = len(all_characters)

    file = "shakespeare.txt"
    with open(file) as f:
        text = f.read().replace("\n", " ")
    text = [all_characters.index(c) for c in text]
    model = MyNet()
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loss_sum = 0
    total = 0
    for i in range(100):
        torch.save(model, "./model.pt")
        with tqdm(total=10000, unit="batch") as pbar:
            for n, (x, y) in enumerate(gen(10000)):
                x = x.cuda()
                y = y.cuda()

                predicts = model(x)

                loss = 0
                for idx in range(y.shape[1]):
                    loss += criterion(predicts[:, idx, :], y[:, idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * x.size(1) 
                total += x.size(1)
                running_loss = loss_sum / total
                pbar.set_postfix({"loss":running_loss})
                pbar.update(1)
                
                if n % 100 == 0:
                    with torch.no_grad():
                        model.eval()
                        print("".join([
                            all_characters[torch.argmax(row).item()]
                            for row in predicts[0][:100]
                        ]))
                        model.train()