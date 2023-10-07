
import torch;
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

# reading all the name 
words = open('names.txt', 'r').read().splitlines()
vocab = sorted(set(''.join(words)))

# encoder and decoder alphabet base.
itos = { idx + 1: word for idx, word in enumerate(vocab) }
itos[0] = '.' # adding dot '.' for boundry demarcation.
stoi = {word: idx for idx, word in itos.items()}
vocab_sz = len(stoi)


gen = torch.Generator().manual_seed(1024)

block_sz = 3  # how many character we want to take in context to predict the next character. 

def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * block_sz
        for alpha in word + '.':
            ix = stoi[alpha]
            X.append(context)
            context = context[1:] + [ix]
            Y.append(ix)

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    return X, Y

random.seed(1599)
random.shuffle(words)

n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)

Xtr, Ytr = build_dataset(words[:n1]) # 80% of the data
Xdev, Ydev = build_dataset(words[n1:n2])  # 10% of the data
Xtes, Ytes = build_dataset(words[n2:])  # 10% of the data


class Linear:
	def __init__(self, fan_in, fan_out, bias=False):
		self.weight = torch.randn((fan_in, fan_out), generator=gen) / (fan_in ** 0.5)
		self.bias = torch.zeros(fan_out) if bias else None

	def __call__(self, inp):
		self.out = inp @ self.weight 
		if self.bias:
			self.out += self.bias
		return self.out

	def parameter(self):
		return [self.weight] + ([self.bias] if self.bias else [])

			

class BatchNorm:

	def __init__(self, dim, eps=1e-5, momentum=0.1):
		self.training = True
		self.momentum = momentum
		self.eps = eps
		
		# batch norm scale and shift
		self.gamma = torch.ones(dim)
		self.beta = torch.zeros(dim)
 
		# batch norm buffer.
		self.run_mean = torch.zeros(dim)
		self.run_var = torch.ones(dim)
	
	def __call__(self, inp):
		
		if self.training:
			xmean = inp.mean(0, keepdim=True)
			xvar = inp.var(0, keepdim=True)
		else:
			xmean = self.run_mean
			xvar = self.run_var

		xhat = (inp - xmean) / torch.sqrt(xvar + self.eps)
		self.out = self.gamma * xhat + self.beta

		if self.training:
			with torch.no_grad():
				self.run_mean =   (1 - self.momentum) *  self.run_mean + xmean * self.momentum
				self.run_var =  ( 1 - self.momentum) * self.run_var + xvar * self.momentum
		
		return self.out
	
	def parameter(self):
		return [self.gamma, self.beta]
	
class Tanh:
	
	def __call__(self, inp):
		self.out =  torch.tanh(inp)
		return self.out
	
	def parameter(self):
		return []

# hyper parameter.
no_emb = 10
hdn_layer = 100
batch_sz = 32

layers = [
	Linear(no_emb*block_sz, hdn_layer), BatchNorm(hdn_layer), Tanh(),
	Linear(hdn_layer, hdn_layer), BatchNorm(hdn_layer), Tanh(),
	Linear(hdn_layer, hdn_layer), BatchNorm(hdn_layer), Tanh(),
	Linear(hdn_layer, hdn_layer), BatchNorm(hdn_layer), Tanh(),
	Linear(hdn_layer, hdn_layer), BatchNorm(hdn_layer), Tanh(),
	Linear(hdn_layer, hdn_layer), BatchNorm(hdn_layer), Tanh(),
	Linear(hdn_layer, vocab_sz)
	]


C = torch.randn((vocab_sz, no_emb), generator=gen)

parameters = [C] + [p for layer in layers for p in layer.parameter()]


for param in parameters:
	param.requires_grad = True

lossi = list()
steps = 250000

with torch.no_grad():
	layers[-1].weight *= 0.1

for i in range(steps):
	batch = torch.randint(0, Xtr.shape[0], (batch_sz, ), generator=gen)
	Xb, Yb = Xtr[batch], Ytr[batch]

	emb = C[Xb]
	x = emb.view(emb.shape[0], -1)

	for layer in layers:
		x = layer(x)

	loss = F.cross_entropy(x, Yb)

	for layer in layers:
		layer.out.retain_grad()

	for param in parameters:
		param.grad = None

	loss.backward()

	lr = 0.1 if i < 20000 else 0.01 
	for param in parameters:
		param.data += -lr * param.grad

	lossi.append(loss.log10().item())

	if i % 25000 == 0:
		print(f"Loss for {i} / {steps}: {loss.item()}")

print(f"Net Loss {loss.item()}")

for _ in range(100):
	out = list()
	context = [0] * block_sz
	while True:
		pemb = torch.tensor([context])
		inferEmb = C[pemb]
		x = inferEmb.view(inferEmb.shape[0], -1)

		for layer in layers:
			layer.training = False
			x = layer(x)

		probs = F.softmax(x, dim=1)
		ix = torch.multinomial(probs, num_samples=1, generator=gen).item()
		context = context[1:] + [ix]
		out.append(itos[ix])
		if itos[ix] == '.':
			break

	print(''.join(out))

plt.figure(figsize=(20, 4));
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        print(f"layer %d (%1s): mean %+.2f. std %.2f, saturated: %.2f%%" % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100));
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} {layer.__class__.__name__}")
plt.legend(legends)
plt.title("Activation Distribution")
plt.show()




