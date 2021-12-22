import numpy as np
import matplotlib.pyplot as plt

wake_thresholds_cv = \
    [[0.047435885604999325, 0.04681841143549723, 0.04681841143549723, 0.04753465448756651, 0.04681841143549723],
    [0.05489245720903364, 0.052097390070719035, 0.048012291945489984, 0.047952251848655335, 0.04091712151746058],
    [0.028724273350926362, 0.02867043083688716, 0.02980213849059867, 0.032375471022701725, 0.02867043083688716],
    [0.02886470083325493, 0.028303506476773013, 0.029505788161997604, 0.02890464731938531, 0.028909023134284707],
    [0.02410930001629276, 0.029369897900065375, 0.024134139474293183, 0.029369496640963017, 0.029369496640963017],
    [0.026760492149805172, 0.02527759348279399, 0.022806095704442017, 0.02527759348279399, 0.02309497845300455],
    [0.02281418455480953, 0.022814791457772262, 0.02281418455480953, 0.02349614785510164, 0.02281418455480953],
    [0.0314017930770945, 0.0392399860732708, 0.03926064882255858, 0.0340419921301857, 0.03532088957518265]]

sws_thresholds_cv = \
    [[0.0032247350686496383, 0.0032247350686496383, 0.0032247350686496383, 0.00321235509934502, 0.0032247350686496383],
    [0.003076212567970452, 0.003076212567970452, 0.003076212567970452, 0.002952727000198832, 0.0028612074034847124],
    [0.003544262192413134, 0.0034899355418059872, 0.0034899355418059872, 0.0033228784248744746, 0.0034899355418059872],
    [0.0024511213706001267, 0.002454450244444325, 0.002454450244444325, 0.002454450244444325, 0.002458845407736408],
    [0.0030587219461970373, 0.0021460416796901835, 0.0021456393742798688, 0.0021456393742798688, 0.0021456393742798688],
    [0.001551214810615064, 0.001551214810615064, 0.001551214810615064, 0.001551214810615064, 0.0016573038745954784],
    [0.0022156063224083805, 0.0022162143810291897, 0.0022156063224083805, 0.002179045195485648, 0.0022156063224083805],
    [0.003968117590477463, 0.003968117590477463, 0.003988798956718255, 0.0021735817822515885, 0.003968117590477463]]

wake_thresholds = np.mean(wake_thresholds_cv, axis=1)
sws_thresholds = np.mean(sws_thresholds_cv, axis=1)

#%%
#gamma_delta_sws_thresholds = [0.00354,0.00287,0.00393,0.00232,0.00193,0.00183,0.00219,0.00222, 0.002229339]
#gamma_delta_wake_thresholds = [0.01810,0.04453,0.02996,0.02366,0.02208,0.02447,0.02672,0.03462, 0.02835651]
#gamma_thetadelta_wake_thresholds = [0.01631,0.03571,0.02362,0.01838,0.01747,0.01976,0.02109,0.03093, 0.0230991156]
#gamma_thetadelta_sws_thresholds = [0.00314,0.00228,0.00329,0.00188,0.00169,0.00172,0.00153,0.00202, 0.0019714286]

gamma_thetadelta_sws_thresholds = [0.00321289, 0.00272605, 0.00307675, 0.00202327, 0.0022235, 0.00163535, 0.00177181, 0.00297665, 0.003190831]
gamma_thetadelta_wake_thresholds = [0.0375733 , 0.03702593, 0.02604808, 0.02116837, 0.01828222, 0.021561  , 0.01746377, 0.02717756, 0.022055037]
gamma_delta_wake_thresholds = [0.04708515, 0.0487743 , 0.02964855, 0.02889753, 0.02727047, 0.02464335, 0.0229507 , 0.03585306, 0.031400108]
gamma_delta_sws_thresholds = [0.00322226, 0.00300851, 0.00346739, 0.00245466, 0.00232834, 0.00157243, 0.00220842, 0.00361335, 0.003966432]

#%%
plt.rc('font', size=12, family='sans-serif')          # controls default text sizes
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

plt.close('all')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
ax1.grid(which='major')
ax2.grid(which='major')

#plt.suptitle('Thresholds per age category')
ax1.set_title('Gamma/delta ratio')
ax1.scatter(range(0, len(gamma_delta_sws_thresholds)), gamma_delta_sws_thresholds,
            marker='+', label = 'Wake threshold', color='crimson')
ax1.scatter(range(0, len(gamma_delta_wake_thresholds)), gamma_delta_wake_thresholds,
            marker='1', label = 'SWS threshold', color='darkcyan')
ax1.set_ylim(0, 0.05)
ax1.tick_params(axis='x', rotation=45)
ax1.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
ax1.set_xticklabels(labels=['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years',
                          '9-13 years', '13-18 years', '0-18 years']) #], rotation='vertical')

ax2.set_title('Gamma/(theta+delta) ratio')
ax2.scatter(range(0, len(gamma_thetadelta_sws_thresholds)), gamma_thetadelta_sws_thresholds,
         marker='+', label='Wake threshold', color='crimson')
ax2.scatter(range(0, len(gamma_thetadelta_wake_thresholds)), gamma_thetadelta_wake_thresholds,
            marker='1', label='SWS threshold', color='darkcyan')
ax1.set_ylabel('Ratio', fontsize=10)
ax2.set_ylim(0, 0.05)
ax2.tick_params(axis='x', rotation=45)
ax2.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
ax2.set_xticklabels(labels=['0-2 months', '2-6 months', '6-12 months', '1-3 years', '3-5 years', '5-9 years',
                          '9-13 years', '13-18 years', '0-18 years']) #], rotation='vertical')

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)

