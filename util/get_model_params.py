import numpy as np
import asdf

#model_name = "12actuators_strehl_200hz_random_wind_speed_10.2"
model_name = '12actuators_strehl_random_wind_33.6.3'
#tree = asdf.open('../../Models/12actuators_strehl_dm_lag=0_99.2_gamma=0_args.asdf').tree
#tree = asdf.open('../../Models/12actuators_strehl_33_3e3photons_r0=0.2_v=10_lag=0_args.asdf').tree
#tree = asdf.open('../../Models/12actuators_strehl_1e3photons_r0=0.2_v=10_lag=0_args.asdf').tree
tree = asdf.open('../../Models/'+model_name+'_args.asdf').tree


for key,values in tree.items():
    print(key,values)
