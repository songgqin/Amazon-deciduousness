"""Spectral unmixing algorithms.

"""

# In[] Imports
import sys
import warnings

import numpy as np
import numpy.linalg as lin

import scipy as sp
import scipy.linalg as splin
import scipy.sparse.linalg as slin

import matplotlib.pyplot as plt

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
            print("... Select the projective proj.")

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
            print("... Select proj. to R-1")

        d = R
        Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:d]

        x_p = sp.dot(Ud.T,Y)
        Yp =  sp.dot(Ud,x_p[:d,:])

        x =  sp.dot(Ud.T,Y)
        u = sp.mean(x,axis=1,keepdims=True)
        y =  x / sum( x * u)

    indice = sp.zeros((R),dtype=int)
    A = sp.zeros((R,R))
    A[-1,0] = 1

    for i in range(R):
        w = sp.random.rand(R,1);
        f = w - sp.dot(A,sp.dot(splin.pinv(A),w))
        f = f / sp.sqrt(sum(f**2))

        v = sp.dot(f.T,y)

        indice[i] = sp.argmax(sp.absolute(v))
        A[:,i] = y[:,indice[i]]

    Ae = Yp[:,indice]

    return (Ae,indice,Yp)

def sunsal(M,y,**kwargs):

    LM,p = M.shape

    L,N = y.shape
    if (LM != L):
        raise ValueError('mixing matrix M and data set y are inconsistent')

    AL_iters = 1000

    Lambda = 0.0

    verbose = True

    positivity = False

    addone = False

    tol = 1e-4;

    x0 = 0;

    for key in kwargs:
        Ukey = key.upper()

        if(Ukey == 'AL_ITERS'):
            AL_iters = np.round(kwargs[key]);
            if (AL_iters < 0 ):
                raise ValueError('AL_iters must a positive integer');
        elif(Ukey == 'LAMBDA_P'):
            Lambda = kwargs[key]
            if (np.sum(np.sum(Lambda < 0)) >  0 ):
                raise ValueError('lambda_p must be positive');
        elif (Ukey =='POSITIVITY'):
            positivity =  kwargs[key]
        elif (Ukey=='ADDONE'):
            addone = kwargs[key]
        elif(Ukey=='TOL'):
            tol = kwargs[key]
        elif(Ukey=='VERBOSE'):
            verbose = kwargs[key]
        elif(Ukey=='X0'):
            x0 = kwargs[key]
            if (x0.shape[0] != p) | (x0.shape[1] != N):
                raise ValueError('initial X is  inconsistent with M or Y');
        elif(Ukey=='X_SOL'):
            X_sol = kwargs[key]
        elif(Ukey=='CONV_THRE'):
            conv_thre = kwargs[key]
        else:

            raise ValueError('Unrecognized option: {}'.format(key))

    Nlambda = np.array(Lambda).size
    if (Nlambda == 1):

        Lambda = Lambda*np.ones((p,N));
    elif (Nlambda != N):
        raise ValueError('Lambda size is inconsistent with the size of the data set');
    else :

        Lambda = np.repeat(Lambda[np.newaxis,:],p,axis=0)

    norm_m = np.sqrt(np.mean(M**2))*(25+p)/p

    M = M/norm_m;
    y = y/norm_m;
    Lambda = Lambda/norm_m**2

    if np.sum(Lambda == 0) and not addone and not positivity :
        z = lin.pinv(M).dot(y)

        res_p = 0
        res_d = 0
        i = 0
        return (z,res_p,res_d,i)

    SMALL = 1e-12
    B = np.ones((1,p))
    a = np.ones((1,N))
    if np.sum(Lambda == 0) and addone and not positivity :
        F = np.transpose(M).dot(M)

        if lin.cond(F) > SMALL :

            IF = lin.inv(F)
            z = IF.dot(M.T).dot(y)-IF.dot(B.T).dot(lin.inv(B.dot(IF).dot(B.T))).dot(B.dot(IF).dot(M.T).dot(y)-a)

            res_p = 0
            res_d = 0
            i = 0
            return (z,res_p,res_d,i)

    mu_AL = 0.01
    mu = 10*np.mean(Lambda) + mu_AL

    UF,sF,VF = lin.svd(M.T.dot(M))
    SF = np.diag(sF)
    IF = UF.dot(np.diag(1/(sF+mu))).dot(UF.T)

    Aux = IF.dot(B.T).dot(lin.inv(B.dot(IF).dot(B.T)))
    x_aux = Aux.dot(a)
    IF1 = IF-Aux.dot(B).dot(IF)
    yy = M.T.dot(y)

    if x0 == 0:
       x = IF.dot(M.T).dot(y)
    else:
        x = x0

    z = x

    d  = 0*z

    tol1 = np.sqrt(N*p)*tol
    tol2 = np.sqrt(N*p)*tol
    i=1
    res_p = float("inf")
    res_d = float("inf")
    maskz = np.ones(z.shape)
    mu_changed = 0

    z_err_set=np.ones((1,10))*1e10

    if np.sum(Lambda ==  0)  and not addone:
        while (i <= AL_iters) and ((np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)):

            if i%10 == 1 :
                z0 = z

            z = np.maximum(x-d,0)

            x = IF.dot(yy+mu*(z+d))

            d = d -(x-z)

            if i%10 == 1 :

                res_p = lin.norm(x-z)

                res_d = mu*lin.norm(z-z0)
                if verbose:
                    print(' i = {}, res_p = {}, res_d = {}\n'.format(i,res_p,res_d))

                if res_p > 10*res_d :
                    mu = mu*2
                    d = d/2
                    mu_changed = 1
                elif res_d > 10*res_p :
                    mu = mu/2
                    d = d*2
                    mu_changed = 1

                if  mu_changed :

                    IF = UF.dot(np.diag(1/(sF+mu))).dot(UF.T)
                    Aux = IF.dot(B.T).dot(lin.inv(B.dot(IF).dot(B.T)))
                    x_aux = Aux.dot(a)
                    IF1 = IF-Aux.dot(B).dot(IF)
                    mu_changed = 0

            i=i+1;

    elif np.sum(Lambda ==  0)  and addone:
        while (i <= AL_iters) and ((np.abs(res_p) > tol1) or (np.abs(res_d) > tol2))  :

            if i%10 == 1 :
                z0 = z

            z = np.maximum(x-d,0)

            x = IF1.dot(yy+mu*(z+d))+x_aux

            d = d -(x-z)

            if i%10 == 1:

                res_p = lin.norm(x-z)

                res_d = mu*lin.norm(z-z0)
                if verbose:
                    print(' i = {}, res_p = {}, res_d = {}\n'.format(i,res_p,res_d))

                if res_p > 10*res_d :
                    mu = mu*2
                    d = d/2
                    mu_changed = 1
                elif res_d > 10*res_p :
                    mu = mu/2
                    d = d*2
                    mu_changed = 1

                if  mu_changed:

                    IF = UF.dot(np.diag(1./(sF+mu))).dot(UF.T)
                    Aux = IF.dot(B.T).dot(lin.inv(B.dot(IF).dot(B.T)))
                    x_aux = Aux.dot(a)
                    IF1 = IF-Aux.dot(B).dot(IF)
                    mu_changed = 0

            i=i+1;

    else :
        softthresh = lambda x,th : np.sign(x)*np.maximum(np.abs(x)-th,0)

        while (i <= AL_iters) and ((np.abs(res_p) > tol1) or (np.abs(res_d) > tol2)) :

            if i%10 == 1:
                z0 = z

            z =  softthresh(x-d,Lambda/mu)

            if positivity :
                z = np.max(z,0)

            if addone :
                x = IF1.dot(yy+mu*(z+d))+x_aux
            else:
                x = IF.dot(yy+mu*(z+d))

            d = d -(x-z)

            if i%10 == 1 :

                res_p = lin.norm(x-z)

                res_d = mu*lin.norm(z-z0)
                if verbose:
                    print(' i = {}, res_p = {}, res_d = {}\n'.format(i,res_p,res_d))

                if res_p > 10*res_d :
                    mu = mu*2
                    d = d/2
                    mu_changed = 1
                elif res_d > 10*res_p :
                    mu = mu/2
                    d = d*2
                    mu_changed = 1

                if mu_changed:

                    IF = UF.dot(np.diag(1./(sF+mu))).dot(UF.T)
                    Aux = IF.dot(B.T).dot(lin.inv(B.dot(IF).dot(B.T)))
                    x_aux = Aux.dot(a)
                    IF1 = IF-Aux.dot(B).dot(IF)
                    mu_changed = 0

            i=i+1

    return (x,res_p,res_d,i)

def soft_neg(y,tau) :

    z = np.maximum(np.abs(y+tau/2) - tau/2, 0)
    z = z*(y+tau/2)/(z+tau/2)
    return z

def sisal(Y,p,**kwargs):

    L,N = Y.shape
    if (L<p) :
        raise ValueError('Insufficient number of columns in y')

    MMiters = 80
    spherize = True

    verbose = 1

    tau = 1

    mu = p*1000/N

    M = 0

    tol_f = 1e-2

    slack = 1e-3

    energy_decreasing = 0

    f_val_back = float("inf")

    lam_sphe = 1e-8

    lam_quad = 1e-6

    AL_iters = 4

    flaged = 0

    for key in kwargs:
        Ukey = key.upper()

        if(Ukey == 'MM_ITERS'):
            MMiters = kwargs[key]
        elif(Ukey == 'SPHERIZE'):
            spherize = kwargs[key]
        elif (Ukey =='MU'):
            mu =  kwargs[key]
        elif (Ukey=='TAU'):
            tau = kwargs[key]
        elif(Ukey=='TOLF'):
            tol_f = kwargs[key]
        elif(Ukey=='M0'):
            M = kwargs[key]
        elif(Ukey=='VERBOSE'):
            verbose = kwargs[key]
        else:

            raise ValueError('Unrecognized option: {}'.format(key))

    if (verbose == 3) or (verbose == 4):
        warnings.filterwarnings("ignore")
    else :
        warnings.filterwarnings("always")

    my = np.mean(Y,axis=1)
    My = np.repeat(my[:,np.newaxis],N,axis=1)
    Myp = np.repeat(my[:,np.newaxis],p,axis=1)

    Y = Y-My
    Up,d,_ = lin.svd(Y@Y.T/N)
    sort_ind = np.argsort(d)[::-1]
    Up = Up[:,sort_ind[:p-1]]
    d = d[sort_ind[:p-1]]

    Y = Up@Up.T@Y

    Y = Y + My

    my_ortho = my-Up@Up.T.dot(my)

    Up = np.concatenate((Up, (my_ortho/np.sqrt(np.sum(my_ortho**2)))[:,np.newaxis] ),axis=1)
    sing_values = d

    Y = Up.T@Y

    if spherize:
        Y = Up@Y
        Y = Y-My
        C = np.diag(1/np.sqrt(d+lam_sphe))
        IC = lin.inv(C)
        Y=C.dot(np.transpose(Up[:,:p-1])).dot(Y)

        Y = np.concatenate((Y,np.ones((1,N))),axis=0)

        Y = Y/np.sqrt(p)

    if M == 0:

        Mvca,_,_ = vca(Y,p,verbose=False)
        M = Mvca

        Ym = np.mean(M,axis=1)
        Ym = np.repeat(Ym[:,np.newaxis],p,axis=1)
        dQ = M - Ym

        M = M + p*dQ
    else:

        M = M-Myp
        M = Up[:,:p-1]@Up[:,:p-1].T@M
        M = M + Myp
        M = Up.T@M

        if spherize:
            M = Up@M-Myp
            M=C@Up[:,:p-1].T@M

            M[p-1,:] = 1

            M = M/np.sqrt(p)

    Q0 = lin.inv(M)
    Q=Q0

    if verbose == 2 or verbose == 4 :

        M = lin.inv(Q)
        fig,ax = plt.subplots()

        line1 = ax.plot(Y[0,:],Y[1,:],'.')
        line2 = ax.plot(M[0,:], M[1,:],'ok')

        ax.set_title('SISAL: Endmember Evolution')

    AAT = np.kron(Y@Y.T,np.eye(p))
    B = np.kron(np.eye(p),np.ones((1,p)))
    qm = np.sum(lin.inv(Y@Y.T)@Y,axis=1)

    H = lam_quad*np.eye(p**2)
    F = H+mu*AAT
    IF = lin.inv(F)

    G = IF@B.T@lin.inv(B@IF@B.T)
    qm_aux = G.dot(qm)
    G = IF-G@B@IF

    Z = Q@Y
    Bk = 0*Z

    hinge = lambda x: np.maximum(-x,0)

    for k in range(MMiters):

        IQ = lin.inv(Q)
        g = -IQ.T
        g = g.flatten(order='C')

        baux = H@Q.flatten(order='C')-g

        q0 = Q.flatten(order='C')
        Q0 = Q

        if verbose == 1 :
            if spherize:

                M = IQ*np.sqrt(p)

                M = M[:p-1,:]

                M = Up[:,:p-1].dot(IC).dot(M)

                M = M + Myp
                M = Up.T.dot(M)
            else:
                M = IQ

            print('\n iter = {0}, simplex volume = {1:.4e}  \n'.format(k, 1/np.abs(lin.det(M))))

        if k == MMiters :
            AL_iters = 100

        while 1 :
            q = Q.flatten(order='C')

            f0_val = -np.log(np.abs(lin.det(Q)))+ tau*np.sum(hinge(Q@Y))
            f0_quad = (q-q0).T.dot(g)+0.5*(q-q0).T.dot(H).dot(q-q0) + tau*np.sum(hinge(Q.dot(Y)))
            for i in range(AL_iters-1):

                dq_aux= Z+Bk
                dtz_b = dq_aux@Y.T
                dtz_b = dtz_b.flatten(order='C')
                b = baux+mu*dtz_b
                q = G.dot(b)+qm_aux
                Q = np.reshape(q,(p,p),order='C')

                Z = soft_neg(Q@Y-Bk,tau/mu);

                Bk = Bk - (Q@Y-Z)
                if verbose == 3 or  verbose == 4 :
                    print('\n ||Q*Y-Z|| = {0:.4f}'.format(lin.norm(Q.dot(Y)-Z)))

                if verbose == 2 or verbose == 4:
                    M = lin.inv(Q)
                    line2.set_xdata(M[0,:])
                    line2.set_ydata(M[1,:])
                    plt.draw()
                    if ~flaged :
                         line3 = ax.plot(M[0,:], M[1,:],'.r')
                         plt.legend('data points', 'M(0)', 'M(k)')
                         flaged = 1

            f_quad = (q-q0).T.dot(g)+0.5*(q-q0).T.dot(H).dot(q-q0) + tau*np.sum(hinge(Q@Y))
            if verbose == 3 or  verbose == 4:
                print('\n MMiter = {0}, AL_iter, = {1},  f0 = {2:2.4f}, f_quad = {3:2.4f},  \n'.format(k,i, f0_quad,f_quad))

            f_val = -np.log(np.abs(lin.det(Q)))+ tau*np.sum(hinge(Q.dot(Y)))
            if f0_quad >= f_quad:
                try:
                    while  f0_val < f_val :
                        if verbose == 3 or  verbose == 4 :
                            print('\n line search, MMiter = {0}, AL_iter, = {1},  f0 = {2:2.4f}, f_val = {3:2.4f},  \n'.format(k,i, f0_val,f_val))

                        Q = (Q+Q0)/2
                        f_val = -np.log(np.abs(lin.det(Q)))+ tau*sum(hinge(Q@Y))
                    break
                except:
                    1+1

    if verbose == 2 or verbose == 4:

        ax.legend('data points','M(0)',  'M(final)')

    if spherize :
        M = lin.inv(Q)

        M = M*np.sqrt(p)

        M = M[:p-1,:]

        M = Up[:,:p-1].dot(IC).dot(M)

        M = M + Myp
    else :
        M = Up.dot(lin.inv(Q))

    return (M,Up,my,sing_values)
