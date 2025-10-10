import numpy as np
import matplotlib.pyplot as plt


for m in [3,10,25,50]:
	wfs = np.loadtxt('{}{}{}'.format('histotripsy',m,'.csv'),delimiter=',')

	idxs = np.argwhere(wfs[:,0]==-1000)
	print(m)
	for n in range(0,len(idxs)):
		print(n,len(idxs))
		if n<(len(idxs)-1):
			idx1,idx2 = idxs[n][0]+1,idxs[n+1][0]
			t,r = wfs[idx1:idx2,0], wfs[idx1:idx2,1]
			print(t)
			#~ t -= t[0]
			#~ t = 2*t/np.max(t)-1
			#~ r = r/np.max(r)
			plt.plot(t,r,'o-')
		else:
			
			idx1 = idxs[n][0]+1
			t,r = wfs[idx1:,0], wfs[idx1:,1]
			#~ t -= t[0]
			#~ t = 2*t/np.max(t)-1
			#~ r = r/np.max(r)
			plt.plot(t,r,'o-')
			
		
	plt.show()
