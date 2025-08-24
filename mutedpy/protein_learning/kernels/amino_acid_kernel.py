import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
"""

FASTA code:

A 	Alanine
B 	Aspartic acid (D) or Asparagine (N)
C 	Cysteine
D 	Aspartic acid
E 	Glutamic acid
F 	Phenylalanine
G 	Glycine
H 	Histidine
I 	Isoleucine
J 	Leucine (L) or Isoleucine (I)
K 	Lysine
L 	Leucine
M 	Methionine/Start codon
N 	Asparagine
O 	Pyrrolysine
P 	Proline
Q 	Glutamine
R 	Arginine
S 	Serine
T 	Threonine
U 	Selenocysteine
V 	Valine
W 	Tryptophan
Y 	Tyrosine
Z 	Glutamic acid (E) or Glutamine (Q) 

"""


def amino_acid_kernel(x, y):
	"""
	one hot:
		negative
		positive
		hydrophobic
		aromatic
		polar
		small
		medium
		large

	distance:
		hamming distance

	:param x:
	:param y:
	:return:
	"""
	f_ord = lambda x: ord(x) - 65
	f_char = lambda x: chr(x + 65)
	ham = lambda x,y: scipy.spatial.distance.hamming(x,y)

	Neg = ['D', 'E']
	Pos = ['R', 'K', 'H']
	Hydro = ['M', 'L', 'I', 'V', 'A']
	Aro = ['F', 'W', 'Y']
	Polar = ['N', 'Q', 'S', 'T']
	Small = ['A', 'S', 'T', 'P', 'G', 'V']
	Medium = ['M', 'L', 'I', 'C', 'N', 'Q', 'K', 'D', 'E']
	Large = ['R', 'H', 'W', 'F', 'Y']
	Special = ['P','C','G']
	Random = ['F','W','L','S','D']

	(n, d) = x.shape
	(m, d) = y.shape
	#print (x.shape,y.shape)
	f = 11
	#K = np.zeros(n, m)
	Phi1 = np.zeros(shape = (n, f * d))
	Phi2 = np.zeros(shape = (m, f * d))
	H = np.zeros(shape = (n, m))

	for i in range(n):
		for j in range(m):
			H[i,j] = ham(x[i],y[j])


	for i in range(n):
		# negative
		for j in range(d):
			Phi1[i, j * f]     = 1 if (f_char(x[i, j]) in Neg) else 0
			Phi1[i, j * f + 1] = 1 if (f_char(x[i, j]) in Pos) else 0
			Phi1[i, j * f + 2] = 1 if (f_char(x[i, j]) in Hydro) else 0
			Phi1[i, j * f + 3] = 1 if (f_char(x[i, j]) in Aro) else 0
			Phi1[i, j * f + 4] = 1 if (f_char(x[i, j]) in Polar) else 0
			Phi1[i, j * f + 5] = 1 if (f_char(x[i, j]) in Small) else 0
			Phi1[i, j * f + 6] = 1 if (f_char(x[i, j]) in Medium) else 0
			Phi1[i, j * f + 7] = 1 if (f_char(x[i, j]) in Large) else 0
			Phi1[i, j * f + 8] = 1 if (f_char(x[i, j]) in Special) else 0
			Phi1[i, j * f + 9] = 1
			Phi1[i, j * f + 10] = 1 if (f_char(x[i, j]) in Random) else 0


	for i in range(m):
		# negative
		for j in range(d):
			Phi2[i, j * f    ] = 1 if (f_char(y[i, j]) in Neg) else 0
			Phi2[i, j * f + 1] = 1 if (f_char(y[i, j]) in Pos) else 0
			Phi2[i, j * f + 2] = 1 if (f_char(y[i, j]) in Hydro) else 0
			Phi2[i, j * f + 3] = 1. if (f_char(y[i, j]) in Aro) else 0
			Phi2[i, j * f + 4] = 1 if (f_char(y[i, j]) in Polar) else 0
			Phi2[i, j * f + 5] = 1 if (f_char(y[i, j]) in Small) else 0
			Phi2[i, j * f + 6] = 1 if (f_char(y[i, j]) in Medium) else 0
			Phi2[i, j * f + 7] = 1 if (f_char(y[i, j]) in Large) else 0
			Phi2[i, j * f + 8] = 1 if (f_char(y[i, j]) in Special) else 0
			Phi2[i, j * f + 9] = 1
			Phi2[i, j * f + 10] = 1 if (f_char(y[i, j]) in Random) else 0

	K = 0.1*((np.dot(Phi1, Phi2.T) - 0.0005*H ) /f).T

	return K

if __name__ == "__main__":
	from protein_benchmark import ProteinBenchmark
	from gauss_procc import GaussianProcess
	from kernels import KernelFunction
	Benchmark = ProteinBenchmark("protein_data_gb1.h5", dim = 1, ref = ['A','C','C','D'])
	names = Benchmark.data['P1'].values
	Benchmark.self_translate()
	xtest = Benchmark.data['P1'].values.reshape(-1,1)


	# similarity
	ax = plt.imshow(amino_acid_kernel(xtest, xtest))
	plt.colorbar()
	print (names)
	plt.xticks(range(xtest.shape[0]),names,fontsize=18)
	plt.yticks(range(xtest.shape[0]),names,fontsize=18)




	ytest = torch.from_numpy(Benchmark.eval(xtest))
	N = 4
	x = xtest[np.random.randint(0,20,N)]
	y = torch.from_numpy(Benchmark.eval(x))

	# amino_acid_kernel
	amino_acid_kernel_torch = lambda x,y: torch.from_numpy(amino_acid_kernel(x,y))
	kernel_object = KernelFunction(kernel_function = amino_acid_kernel_torch)
	GP = GaussianProcess(kernel_custom = kernel_object, s = 0.01)
	GP.fit_gp(x,y)
	[mu,s] = GP.mean_var(xtest)

	lcb = torch.argmax(mu - 2*s)


	plt.figure(2)
	plt.xlabel("Amino Acids", fontsize = 20)
	plt.ylabel("Fitness", fontsize = 20)
	plt.plot(xtest,xtest*0+(mu[lcb] - 2*s[lcb]).numpy(),'orange',label = "Lower Confidence Bound",alpha = 0.5)
	plt.scatter(xtest,ytest.numpy(),color = "r", label = "Ground Truth",alpha = 0.5, s = 100)
	plt.xticks(xtest,names,fontsize=18)
	plt.yticks(fontsize =18)
	plt.errorbar(xtest,mu.numpy(),yerr = 2*s.numpy(),fmt='go', label = "Estimate with Confidence",alpha = 0.5, ms = 10)
	plt.scatter(x,y,color = "blue",marker = 'o',s = 100, label = "Experiments", alpha = 0.5)
	plt.plot(xtest,xtest*0,"k",alpha = 0.5)
	plt.legend(fontsize = 30)
	#plt.show()



	### discart regions which are suboptimal.

	plt.figure(3)
	plt.xlabel("Amino Acids", fontsize=20)
	plt.ylabel("Fitness", fontsize=20)
	plt.plot(xtest, xtest * 0 + (mu[lcb] - 2 * s[lcb]).numpy(), 'orange', label="Lower Confidence Bound", alpha=0.5, lw = 3)
	plt.scatter(xtest, ytest.numpy(), color="r", label="Ground Truth", alpha=0.5, s=100)
	plt.xticks(xtest, names, fontsize=18)
	plt.yticks(fontsize=18)
	plt.errorbar(xtest, mu.numpy(), yerr=2 * s.numpy(), fmt='go', label="Estimate with Confidence", alpha=0.5, ms=10, lw = 3)
	plt.scatter(x, y, color="blue", marker='o', s=100, label="Experiments", alpha=0.5)
	plt.plot(xtest, xtest * 0, "k", alpha=0.5)

	ucb = mu + 2 * s  # upper confidence estimate (95%)
	lcb_value = mu[lcb] - 2 * s[lcb]
	i = 0
	first = False
	for xz in xtest:
		if (ucb < lcb_value)[i, 0]:
			print (xz)
			if first == False:
				plt.fill_between([xz[0]-0.5,xz[0]+0.5], -0.25, 0.25, alpha=0.4, color='orange', label = "Discarted Regions")
				first = True
			else:
				plt.fill_between([xz[0] - 0.5, xz[0] + 0.5], -0.25, 0.25, alpha=0.4, color='orange')
		i = i + 1

	plt.legend(fontsize=30)


	### identify the action to be taken

	plt.figure(4)
	plt.xlabel("Amino Acids", fontsize=20)
	plt.ylabel("Fitness", fontsize=20)
	plt.plot(xtest, xtest * 0 + (mu[lcb] - 2 * s[lcb]).numpy(), 'orange', label="Lower Confidence Bound", alpha=0.5, lw = 3)
	plt.scatter(xtest, ytest.numpy(), color="r", label="Ground Truth", alpha=0.5, s=100)
	plt.xticks(xtest, names, fontsize=18)
	plt.yticks(fontsize=18)
	plt.errorbar(xtest, mu.numpy(), yerr=2 * s.numpy(), fmt='go', label="Estimate with Confidence", alpha=0.5, ms=10, lw = 3)
	plt.scatter(x, y, color="blue", marker='o', s=100, label="Experiments", alpha=0.5)
	plt.plot(xtest, xtest * 0, "k", alpha=0.5)

	ucb = mu + 2 * s  # upper confidence estimate (95%)
	lcb_value = mu[lcb] - 2 * s[lcb]
	i = 0
	first = False
	for xz in xtest:
		if (ucb < lcb_value)[i, 0]:
			print (xz)
			if first == False:
				plt.fill_between([xz[0]-0.5,xz[0]+0.5], -0.25, 0.25, alpha=0.4, color='orange', label = "Discarted Regions")
				first = True
			else:
				plt.fill_between([xz[0] - 0.5, xz[0] + 0.5], -0.25, 0.25, alpha=0.4, color='orange')
		i = i + 1

	max_variance = torch.argmax(s[(ucb > lcb_value)])

	plt.scatter(xtest[max_variance],(mu+2*s)[max_variance], marker = '*', s = 400, label = "Next Experiment", color = 'orange')
	plt.legend(fontsize=30)
	#plt.show()


	## next step
	# new x

	xnext = xtest[max_variance].reshape(-1,1)
	ynext = torch.from_numpy(Benchmark.eval(xnext)).view(-1,1)

	#(x,y)
	# concatanate
	x = np.concatenate((x,xnext))
	y = torch.cat((y,ynext))
	GP.fit_gp(x, y)
	[mu, s] = GP.mean_var(xtest)

	lcb = torch.argmax(mu - 2 * s)

	plt.figure(5)
	plt.xlabel("Amino Acids", fontsize=20)
	plt.ylabel("Fitness", fontsize=20)
	plt.plot(xtest, xtest * 0 + (mu[lcb] - 2 * s[lcb]).numpy(), 'orange', label="Lower Confidence Bound", alpha=0.5,
			 lw=3)
	plt.scatter(xtest, ytest.numpy(), color="r", label="Ground Truth", alpha=0.5, s=100)
	plt.xticks(xtest, names, fontsize=18)
	plt.yticks(fontsize=18)
	plt.errorbar(xtest, mu.numpy(), yerr=2 * s.numpy(), fmt='go', label="Estimate with Confidence", alpha=0.5, ms=10,
				 lw=3)
	plt.scatter(x, y, color="blue", marker='o', s=100, label="Experiments", alpha=0.5)
	plt.plot(xtest, xtest * 0, "k", alpha=0.5)

	ucb = mu + 2 * s  # upper confidence estimate (95%)
	lcb_value = mu[lcb] - 2 * s[lcb]
	i = 0
	first = False
	for xz in xtest:
		if (ucb < lcb_value)[i, 0]:
			print(xz)
			if first == False:
				plt.fill_between([xz[0] - 0.5, xz[0] + 0.5], -0.25, 0.25, alpha=0.4, color='orange',
								 label="Discarted Regions")
				first = True
			else:
				plt.fill_between([xz[0] - 0.5, xz[0] + 0.5], -0.25, 0.25, alpha=0.4, color='orange')
		i = i + 1

	plt.legend(fontsize=30)


	plt.show()