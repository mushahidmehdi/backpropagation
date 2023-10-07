
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# reading all the name 
words = open('names.txt', 'r').read().splitlines()
vocab = sorted(set(''.join(words)))

# encoder and decoder alphabet base.
itos = { idx + 1: word for idx, word in enumerate(vocab) }
itos[0] = '.' # adding dot '.' for boundry demarcation.
stoi = {word: idx for idx, word in itos.items()}
vocab_sz = len(stoi)

# building the dataset from the names.
blk_sz = 3
def build_dataset(dt, blk_size):
	context = [0] * blk_size
	label, target = [], []
	for name in dt:
		for x in name + '.': # dot (.) represent the word end.
			idx = stoi[x]
			label.append(context)
			target.append(idx)
			context = context[1:] + [idx]

	return torch.tensor(label), torch.tensor(target)

tranSplit = int(len(words) * 0.8)  # 80% of data
devSplit = int(len(words) * 0.15)  # 15% of data
testSplit = int(len(words) * 0.05) # 5% of data

trainData, trainTarget = build_dataset(words[:tranSplit], blk_sz)
devData, devTarget = build_dataset(words[:devSplit], blk_sz)
testData, testTarget = build_dataset(words[:testSplit], blk_sz)

# hyperparameter for the models;
no_emb = 10
no_hdn_lyr = 200
gen = torch.Generator().manual_seed(1024) # for reproducibility.

# initalization layers.
# torch.randn generate a matrix of std=1, mean=0;
W1 = torch.randn((blk_sz * no_emb, no_hdn_lyr), generator=gen) * ((5/3)/((no_emb*blk_sz)**0.5)) # Xavier initialization: prevent saturation/vinishing 
b1 = torch.randn(no_hdn_lyr,					generator=gen)
W2 = torch.randn((no_hdn_lyr, vocab_sz),		generator=gen)
b2 = torch.randn(vocab_sz,						generator=gen)
embedMtx = torch.randn((vocab_sz, no_emb),		generator=gen) # Embedding matrx
parameters = [W1, b1, W2, b2, embedMtx]

for p in parameters:
	p.requires_grad = True

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10 ** lre
lri = list()
lossi = list()
stepsi = list()

batch_size = 64
for i in range(10000):
	batchNorm = torch.randint(0, trainData.shape[0], (batch_size, ), generator=gen)
	batchLabel, batchTarget = trainData[batchNorm], trainTarget[batchNorm]
	embedData = embedMtx[batchLabel]
	hpred = torch.tanh(embedData.view(embedData.shape[0], -1) @ W1 + b1)
	logist = hpred @ W2 + b2
	loss = F.cross_entropy(logist, batchTarget)

	for p in parameters:
		p.grad = None
	loss.backward()

	lr = 0.11 if i < 9000 else 0.01
	for p in parameters:
		p.data += -lr * p.grad


	# lri.append(lr)
	lossi.append(loss.item())
	stepsi.append(i)

print('train loss', loss.item())


devbatNorm = torch.randint(0, devData.shape[0], (batch_size, ), generator=gen)
Xdev, Ydev = devData[devbatNorm], devTarget[devbatNorm]
devEmbd = embedMtx[Xdev]
hpred = torch.tanh(devEmbd.view(devEmbd.shape[0] , -1) @ W1 + b1)
logits = hpred @ W2 + b2
devLoss = F.cross_entropy(logist, Ydev)
print('devLoss', devLoss.item())


# plt.plot(lri, lossi)
# plt.plot(stepsi, lossi)

# plt.plot(lossi)
# plt.show()

