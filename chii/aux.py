#@markdown aux plot funcs
from .tools import *
from .tools import get_c_p_m_star2

def aux_plot_popsizes(trj):
    fig, ax = plt.subplots(figsize=(7,3))
    A = np.asarray(trj['m']).squeeze(-1)
    if A.ndim == 1:
        ax.plot(trj['t'], A, label=rf"$m_1$")
    else:
        for j in range(A.shape[1]):
            ax.plot(trj['t'], A[:, j], label=rf"$m_{j+1}$")
    # ax.legend()
    ax.set_yscale('log')
    # ax.hlines(1, 0, final_time, linestyles='--', colors='red')
    # ax.set_ylim(0,0.001)
    fig.subplots_adjust(right=0.6)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("population density")
    plt.show()

#@markdown aux Plot funcs
def aux_plot_status(params):
    M,S,D,U,T,L,Fn,Ft,Phi,psi,f_n,f_t,_ = params.values()
    cstar, p_star, mstar, a_star, eta = get_c_p_m_star2(params,get_cMiki=True)
    smoothChoice=True
    plt.figure(); padding=1.3

    extent=np.max(np.concatenate((eta,L),axis=1),axis=1)
    plt.axis('equal')

    plt.quiver(0, 0, *L, color='#90f', label='influx', scale=1,angles='xy', scale_units='xy');# plt.scatter(*L,s=0)

    plt.scatter(*eta,marker='*',color='#9f2',s=200,label=r'$\eta$')
    plt.scatter(*cstar,marker='*',color='#f29',s=100,label=r'$\tilde{c}^*$')
    plt.xlim(0,extent[0]*padding)
    plt.ylim(0,extent[1]*padding)

    x_vals2=np.linspace(*plt.xlim(),resolution)
    regimeCurves = -(U[0,:]*Fn[1,:]*np.c_[x_vals2]) / (U[0,:]*np.c_[x_vals2] - U[1,:]*(Fn[0,:] + np.c_[x_vals2]))
    plt.plot(x_vals2, regimeCurves, linewidth=0.2)

    plt.title(r'Strain $\eta$')
    plt.xlabel(r"$\tilde{c}_1$")
    plt.ylabel(r"$\tilde{c}_2$", rotation=90)
    plt.legend(bbox_to_anchor=(0.5, -.3), loc="upper center")

    plt.show()

def aux_plot_traj(params, initState, final_time=10000,plot_concentrations=False, ):
    X0, Y0, m0, P0, R0 =initState
    t_eval = np.linspace(0, final_time, resolution)

    # traj_i = stabilize(X0,Y0,m0,P0,R0,ps)
    # initM = copy_parent_initial_conditions(parent,Y0,m0,P0,R0,initPop=0)
    M,S,D,U,T,L,Fn,Ft,Phi,psi,f_n,f_t,_ = params.values()
    cstar, p_star, mstar, a_star, eta = get_c_p_m_star2(params,get_cMiki=True)

    # traj_i = integrate_system(X0, *initM,         D, U, T, L, Fn, Ft, Phi, psi, (0.0, t_eval[-1]), t_eval=t_eval, optAlloc=True)
    traj_i = integrate_system(X0, Y0, m0, P0, R0, D, U, T, L, Fn, Ft, Phi, psi,(0.0, t_eval[-1]), t_eval=t_eval, optAlloc=True)
    # traj_i['P'][-1].sum(0)+ traj_i['R'][-1].T should give ones
    traj = traj_i

    # --- second plot (m_i) ---
    fig, ax = plt.subplots(figsize=(7,3))
    A = np.asarray(traj['m']).squeeze(-1)
    if A.ndim == 1:
        ax.plot(traj['t'], A, label=rf"$m_1$")
    else:
        for j in range(A.shape[1]):
            ax.plot(traj['t'], A[:, j], label=rf"$m_{j+1}$")
    ax.scatter([final_time]*S,mstar,marker='*',color='#f92',s=200,label=r'$\tilde{m}^*$')

    ax.legend()
    ax.set_yscale('log')
    fig.subplots_adjust(right=0.6)

    return fig, ax

    if plot_concentrations:
        plt.show()
        fig, ax = plt.subplots(figsize=(7,3))
        for name in ['Y','X','R']:
            # for name in ['X']:
            A = np.asarray(traj[name]).squeeze(-1)
            if A.ndim == 1:
                ax.plot(traj['t'], A, label=rf"${name}_1$")
            else:
                for j in range(A.shape[1]):
                    ax.plot(traj['t'], A[:, j], label=rf"${name}_{j+1}$")

        ax.scatter([final_time]*M,cstar,marker='*',color='#f29',s=200,label=r'$\tilde{c}^*$')
        ax.scatter([final_time]*M,a_star,marker='*',color='#f92',s=200,label=r'$\tilde{a}^*$')

        ax.legend()
        ax.set_yscale('log')
        # ax.set_ylim((1e-3,1e-1))
        ax.set_xlabel('time')
        plt.show()

def aux_plot_ZNGIs(parameters, definition=80, maxrange2 = 0.9):
    # visualize-------------------------------------------------
    definition=80
    maxrange2 = 0.9

    def best_lmbd(x, y):
        # AUXILIARY - gives best possible growth rate under a nutrient concentration of TWO nutrients x,y
        # ufc=np.c_[[x,y]]
        # c = Fn/(U/ufc-1)
        c=np.c_[[x,y]]
        critC= U * monod(c,Fn)
        softWin = smoothArgMax(critC,.1)
        d=(critC*softWin).sum(0) /T.T*eps_R/eps_P*eps_C/eps_A
        psi = d/(d+1)
        Phi= (1-psi)*softWin
        psi = psi.reshape(S, 1)
        ps = parameters.copy()
        ps['Phi']=Phi
        ps['psi']=psi
        # return lambda_from_alloc(c, parameters)
        return lambda_from_alloc(c, ps)

    M,S,D,U,T,L,Fn,Ft,Phi,psi,f_n,f_t,_ = parameters.values()
    cstar, p_star, mstar, a_star, eta = get_c_p_m_star2(parameters,get_cMiki=True)

    x_vals2 = np.linspace(0,maxrange2, definition)   # X range
    y_vals2 = np.linspace(0,maxrange2, definition)   # Y range
    X2, Y2 = np.meshgrid(x_vals2, y_vals2, indexing='xy')
    Z2r = [best_lmbd(x,y) for x,y in zip(X2.ravel(),Y2.ravel())]
    Z2 = np.array(Z2r).reshape(definition,definition,S)

    mat = np.sqrt(np.prod(np.abs(Z2),axis=2))**(-1/5)
    regimeCurves = -(U[0,:]*Fn[1,:]*np.c_[x_vals2]) / (U[0,:]*np.c_[x_vals2] - U[1,:]*(Fn[0,:] + np.c_[x_vals2]))

    plt.figure()
    top = np.abs(np.nanmax(Z2))
    im = plt.imshow(mat.astype('float32'), interpolation='none', cmap='afmhot', origin='lower' ,extent=[x_vals2[0], x_vals2[-1], y_vals2[0], y_vals2[-1]],)

    plt.axis('equal')
    plt.plot(x_vals2, regimeCurves)
    plt.quiver(0, 0, *L, color='#90f', label='influx', scale=1, units='xy', width=1e-2);# plt.scatter(*L,s=0)

    plt.scatter(*eta,marker='*',color='#9f2',s=200,label=r'$\eta$')
    plt.scatter(*cstar,marker='*',color='#f29',s=200,label=r'$\tilde{c}^*$')

    plt.title(r'Zero net growth isoclines')
    plt.xlabel(r"$\tilde{c}_1$")
    plt.ylabel(r"$\tilde{c}_2$", rotation=90)
    plt.legend(bbox_to_anchor=(0.5, -.3), loc="upper center")
    plt.xlim((0,maxrange2))
    plt.ylim((0,maxrange2))
    plt.show()
