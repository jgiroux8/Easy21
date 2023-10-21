import numpy as np
import matplotlib.pyplot as plt


def plot_value_function(Q,episodes,method):
    X = range(1,11)
    Y = range(1,22)

    X, Y = np.mgrid[X,Y]

    Z = np.max(Q,2)

    fig = plt.figure(dpi=300,facecolor="w")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, Z,rstride=1, cstride=1,
                         linewidth=0.75, antialiased=True,color='grey')
    ax.set_zlim(-1,1)
    ax.set_xlabel('Dealer Showing',fontsize=14)
    ax.set_ylabel('Player Sum',fontsize=14)
    ax.set_zlabel(r'$V^{\star}(s)$',fontsize=14)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    ax.grid(False)
    fig.savefig('Figures/'+method+'_{0}_episodes.pdf'.format(episodes),bbox_inches='tight')
    
    plt.close()
    
    
def plot_learning_curve(mse_,lambda_values):
    
    colors = ['red','blue','green','purple','black','orange','magenta','yellow','cyan','lime','pink']
    for i,mse in enumerate(mse_):
        x = mse[:,0][1:]
        y = mse[:,1][1:]
        plt.plot(x,y,color=colors[i],label=r'$\lambda$ = {0}'.format(lambda_values[i]),linestyle='--')
    
    
    
    plt.xlabel('Episodes',fontsize=18)
    plt.ylabel(r'$MSE(V^{\star} - V)$',fontsize=18)
    plt.title(r"Sarsa($\lambda$)",fontsize=20)
    plt.legend(loc='best',fontsize=15,ncol=2)
    plt.savefig('Figures/Sarsa_mse.pdf',bbox_inches='tight')
    plt.close()
    

def plot_mse_lambda(mse_,lambda_values):

    x = lambda_values
    y = mse_
    plt.plot(x,y,color='k',linestyle='--',marker='s')
    plt.xlabel(r'$\lambda$',fontsize=18)
    plt.ylabel(r'$MSE(V^{\star} - V)$',fontsize=18)
    plt.title(r"Sarsa($\lambda$)",fontsize=20)
    plt.savefig('Figures/Sarsa_mse_lambda.pdf',bbox_inches='tight')
    plt.close()
