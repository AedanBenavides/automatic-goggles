import torch
import torch.nn as nn
import random

#Bleheehhhhhhhhh
chars = 'abcdefghijklmnopqrstuvwxyz '
char_to_idx = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 16)
        self.fc1 = nn.Linear(16 * 15, 64)  
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 15 * vocab_size) 
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.embed(x)

        x = x.view(x.size(0), -1)  

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        x = self.fc3(x)

        x = x.view(-1, 15, vocab_size)

        return x

def text_to_tensor(text):
    indices = [char_to_idx.get(c, 0) for c in text.lower()]

   
    indices = (indices + [0]*15)[:15]
    return torch.tensor(indices)

def tensor_to_text(t):
    return ''.join([chars[i] for i in t if i < len(chars)])

# make some data
words = ['cat', 'dog', 'sun', 'bat', 'hat', 'rat', 'mat', 'cup', 'car', 'box']
data = []
for _ in range(500):
    w1 = random.choice(words)
    w2 = random.choice(words)
    inp = w1 + ' ' + w2
    out = w1 + w2
    data.append((inp, out))

model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("training...")
for epoch in range(100):
    random.shuffle(data)
    loss_total = 0
    
    for inp_text, out_text in data:
        x = text_to_tensor(inp_text).unsqueeze(0)
        y = text_to_tensor(out_text)
        
        pred = model(x)
        loss = loss_fn(pred[0], y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_total += loss.item()
    
    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss: {loss_total/len(data):.3f}")


print("\ntesting:")
model.eval()
tests = [('cat dog', 'catdog'), ('sun bat', 'sunbat'), ('hat cup', 'hatcup')]
for inp, expected in tests:
    x = text_to_tensor(inp).unsqueeze(0)
    pred = model(x)
    pred_idx = torch.argmax(pred[0], dim=1)
    result = tensor_to_text(pred_idx)
    print(f"{inp} -> {result.strip()} (expected: {expected})")
