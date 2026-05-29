"""Vertex component analysis implementation.

"""

# In[] Imports
import sys
import scipy as sp
import scipy.linalg as splin

# In[] Functions
def estimate_snr(Y,r_m,x):

  [L, N] = Y.shape
  [p, N] = x.shape

  P_y     = sp.sum(Y**2)/float(N)
  P_x     = sp.sum(x**2)/float(N) + sp.sum(r_m**2)
  snr_est = 10*sp.log10( (P_x - p/L*P_y)/(P_y - P_x) )

  return snr_est

def vca(Y,R,verbose = True,snr_input = 0):

  if len(Y.shape)!=2:
    sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

  [L, N]=Y.shape

  R = int(R)
  if (R<0 or R>L):
    sys.exit('ENDMEMBER parameter must be integer between 1 and L')

  if snr_input==0:
    y_m = sp.mean(Y,axis=1,keepdims=True)
    Y_o = Y - y_m
    Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:R]
    x_p = sp.dot(Ud.T, Y_o)

    SNR = estimate_snr(Y,y_m,x_p);

    if verbose:
      print("SNR estimated = {}[dB]".format(SNR))
  else:
    SNR = snr_input
    if verbose:
      print("input SNR = {}[dB]\n".format(SNR))

  SNR_th = 15 + 10*sp.log10(R)

  if SNR < SNR_th:
    if verbose:
      print("... Select proj. to R-1")

      d = R-1
      if snr_input==0:
        Ud = Ud[:,:d]
      else:
        y_m = sp.mean(Y,axis=1,keepdims=True)
        Y_o = Y - y_m

        Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:d]
        x_p =  sp.dot(Ud.T,Y_o)

      Yp =  sp.dot(Ud,x_p[:d,:]) + y_m

      x = x_p[:d,:]
      c = sp.amax(sp.sum(x**2,axis=0))**0.5
      y = sp.vstack(( x, c*sp.ones((1,N)) ))
  else:
    if verbose:
      print("... Select the projective proj.")

    d = R
    Ud  = splin.svd(sp.dot(Y,Y.T)/float(N))[0][:,:d]

    x_p = sp.dot(Ud.T,Y)
    Yp =  sp.dot(Ud,x_p[:d,:])

    x =  sp.dot(Ud.T,Y)
    u = sp.mean(x,axis=1,keepdims=True)
    y =  x / sp.dot(u.T,x)

  indice = sp.zeros((R),dtype=int)
  A = sp.zeros((R,R))
  A[-1,0] = 1

  for i in range(R):
    w = sp.random.rand(R,1);
    f = w - sp.dot(A,sp.dot(splin.pinv(A),w))
    f = f / splin.norm(f)

    v = sp.dot(f.T,y)

    indice[i] = sp.argmax(sp.absolute(v))
    A[:,i] = y[:,indice[i]]

  Ae = Yp[:,indice]

  return Ae,indice,Yp
