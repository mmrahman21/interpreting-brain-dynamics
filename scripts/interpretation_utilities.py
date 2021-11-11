import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def do_t_test(x_HC, x_PT):
    
    t_values = np.zeros(53)
    p_values = np.zeros(53)
    
    for i in range(53):  # For 53 ica components
        
        
        '''it computes the t-statistic and then compares against the critical t-value which is computed internally.
        With this it gets the value of p. If it is low enough <=0.0005, it means we can reject the null hypothesis, 
        which in turn implies that the distributions are statistically different'''
        
        
        a = x_HC[:, i]
        b = x_PT[:, i]
        
#         N = len(a)
        
        # Calculate standard deviation by dividing N - 1 (i.e. using ddof = 1)
#         var_a = a.var(ddof=1)
#         var_b = b.var(ddof=1)
        
        # std deviation 
        
#         s = np.sqrt((var_a + var_b)/2)
        
        # calculate t-statistics
#         t = (a.mean() - b.mean())/(s*np.sqrt(2/N))
        
        
        ## Compare with the critical t-value
        #Degrees of freedom
#         df = 2*N - 2

        #p-value after comparison with the t 
#         p = 1 - stats.t.cdf(t,df=df)

        ## Cross Checking with the internal scipy function
        t2, p2 = stats.ttest_ind(a,b)
        
        t_values[i], p_values[i] = t2, p2
       
    
        
    indices = [1 if p_values[item] < 0.05/53 else 0 for item in range(53) ]
    
#     indices = [1 if p_values[item] < 0.05 and abs(t_values[item]) > 5 else 0 for item in range(53) ]
#     print(f"{indices}\n\n")
    
    return indices

#         print(f"t-computed: {t:7.4f}  p-computed: {p:7.4f}  t-scipy: {t2:7.4f}  p-scipy: {p2:7.4f}\n")
       
        
    

def plot_avg_pattern(x_HC, x_PT, path):
    
    fig =  plt.figure(figsize=[14, 6])

    ax1 = plt.subplot(1, 2, 1)

    HC_avg = np.max(x_HC, axis=0)
    PT_avg = np.max(x_PT, axis=0)

    p1 = ax1.imshow(HC_avg, interpolation='nearest', aspect='auto', cmap='RdBu')
    
    ax1.axes.xaxis.set_ticks([])
    ax1.axes.yaxis.set_ticks([])
    
    
    ax1.set_title("Healthy Controls", fontsize='x-large')
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    
    ax2 = plt.subplot(1, 2, 2)

    p2 = ax2.imshow(PT_avg, interpolation='nearest', aspect='auto', cmap='RdBu')
    
    ax2.axes.xaxis.set_ticks([])
    ax2.axes.yaxis.set_ticks([])
    
    ax2.set_title("Patients", fontsize='x-large')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

   
    fig.savefig(path, format='png', dpi=600)
    print('Plots Saved...', path)
    plt.close(fig)

def plot_pattern(data, masked_sal, sal, labels, d, l, predictions, path):
    
    fig, axes = plt.subplots(10, 3)
    
    for i in range(10):
        axes[i, 0].imshow(data[d[i]], interpolation='nearest', aspect='auto', cmap='Reds')     
        axes[i, 1].imshow(masked_sal[d[i]], interpolation='nearest', aspect='auto', cmap='Reds')
        axes[i, 2].imshow(sal[d[i]], interpolation='nearest', aspect='auto', cmap='Reds')


    # Turn off *all* ticks & spines, not just the ones with colormaps.

    for i in range(10):
        # axes[i,0].set_axis_off()
        # axes[i, 1].set_axis_off()
        axes[i,0].axes.xaxis.set_ticks([])
        axes[i,0].axes.yaxis.set_ticks([])

        axes[i,1].axes.xaxis.set_ticks([])
        axes[i,1].axes.yaxis.set_ticks([])
        
        axes[i,2].axes.xaxis.set_ticks([])
        axes[i,2].axes.yaxis.set_ticks([])

        axes[i,0].set_ylabel(str(int(labels[d[i]])), fontsize=12)

    axes[0, 0].set_title('Data', fontsize='medium')
    axes[0, 1].set_title(f'Fraction of Saliency', fontsize='medium')
    axes[0, 2].set_title(f'Full Saliency', fontsize='medium')
    
    
    fig.savefig(path, transparent=True, bbox_inches='tight', pad_inches=0)
    
    
#     fig.savefig(path, format='png', dpi=600)
    print('Plots Saved...', path)
    plt.close(fig)


    
def plot_binary_mask(data_mask, labels, d, l, predictions, path):
    
    print('Plotting binary mask')
    
    fig, axes = plt.subplots(10, 1)
    
    for i in range(10):
        axes[i].imshow(data_mask[d[i]], interpolation='nearest', aspect='auto')     
#         axes[i, 1].imshow(masked_sal[d[i]], interpolation='nearest', aspect='auto', cmap='Reds')
#         axes[i, 2].imshow(sal[d[i]], interpolation='nearest', aspect='auto', cmap='Reds')


    # Turn off *all* ticks & spines, not just the ones with colormaps.

    for i in range(10):
        # axes[i,0].set_axis_off()
        # axes[i, 1].set_axis_off()
        axes[i].axes.xaxis.set_ticks([])
        axes[i].axes.yaxis.set_ticks([])

#         axes[i,1].axes.xaxis.set_ticks([])
#         axes[i,1].axes.yaxis.set_ticks([])
        
#         axes[i,2].axes.xaxis.set_ticks([])
#         axes[i,2].axes.yaxis.set_ticks([])

        axes[i].set_ylabel(str(int(labels[d[i]])), fontsize=12)

#     axes[0, 0].set_title('Data', fontsize='medium')
#     axes[0, 1].set_title(f'Fraction of Saliency', fontsize='medium')
#     axes[0, 2].set_title(f'Full Saliency', fontsize='medium')
    
    
    fig.savefig(path, transparent=True, bbox_inches='tight', pad_inches=0)
    
    plt.show()
    
    
#     fig.savefig(path, format='png', dpi=600)
    print('Plots Saved...', path)
    plt.close(fig)

    
def plot_dfnc(FinalMatrix, Labels, D, L, predictions, path):
    fig, axes = plt.subplots(4,4)
        
    HC_counter = 0
    SZ_counter = 0

    for i in range(16):

        if Labels[D[i]]==0:
            ax = plt.subplot(4, 4, HC_counter+1)
            HC_counter = HC_counter+1
        else:

            ax = plt.subplot(4, 4, 16-SZ_counter)
            SZ_counter = SZ_counter+1

        ax.imshow(FinalMatrix[D[i]], interpolation='nearest', aspect='equal', cmap='jet')
        ax.set_title(str(int(D[i]))+"/"+str(int(Labels[D[i]]))+"/"+str(int(predictions[D[i], 1])), fontsize='x-small')

        ax.set_xlabel('')
        ax.set_ylabel('')

        plt.tick_params(
        axis='both',          # changes apply to both axes
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top = False,
        left = False,
        right = False,

        labelleft = False,
        labelbottom=False) # labels along the bottom edge are off


    plt.subplots_adjust(hspace = 0.3, wspace = 0.0001)
    fig.savefig(path, format='png', dpi=600)
    print('Plots Saved...', path)
    plt.close(fig)

def plot_average_dfnc(X_HC, X_PT, path):
    
    fig =  plt.figure(figsize=[14, 6])

    ax1 = plt.subplot(1, 2, 1)

    HC_avg = np.sum(X_HC, axis=0)/X_HC.shape[0]
    PT_avg = np.sum(X_PT, axis=0)/X_PT.shape[0]

    p1 = ax1.imshow(HC_avg, interpolation='nearest', aspect='equal', cmap='jet')
    ax1.set_title("Healthy Controls", fontsize='x-large')
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    fig.colorbar(p1, ax=ax1)

    plt.tick_params(
    axis='both',          # changes apply to both axes
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top = False,
    left = False,
    right = False,

    labelleft = False,
    labelbottom=False) # labels along the bottom edge are off


    ax2 = plt.subplot(1, 2, 2)

    p2 = ax2.imshow(PT_avg, interpolation='nearest', aspect='equal', cmap='jet')
    ax2.set_title("Patients", fontsize='x-large')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    fig.colorbar(p2, ax=ax2)


    plt.tick_params(
    axis='both',          # changes apply to both axes
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top = False,
    left = False,
    right = False,

    labelleft = False,
    labelbottom=False) # labels along the bottom edge are off


    plt.subplots_adjust(hspace = 0.3, wspace = 0.0001)
    fig.savefig(path, format='png', dpi=600)
    print('Plots Saved...', path)
    plt.close(fig)