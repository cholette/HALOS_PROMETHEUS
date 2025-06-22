""" Demo of clean vs soiled heliostat field """
import inputs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sp_module

case_filename = "./../case_inputs/flat_50_ca_case.csv"
case_name = "flat_daggett_50"
filenames = inputs.readCaseFile(case_filename)
settings = inputs.readSettingsFile(filenames["settings"])
N = pd.read_csv(filenames['field_filename']).shape[0]
tower_height = float(pd.read_csv(filenames['receiver_filename'],index_col=0,names=['name','value']).loc['tow_height'].value)
soiling = [1.0]*N
for ii in range(200,500):
    soiling[ii] = 0.75  # Example soiling condition for the first heliostat

# %% Clean
aimpoints=None
sp_flux = sp_module.SP_Flux(filenames,use_sp_field=False)
outputs_clean = sp_flux.run_sp_case(case_name=case_name,
                                    sp_aimpoint_heur = False, 
                                    saveCSV = True,
                                    hour_id = settings["hour_idx"], 
                                    weather_data = filenames["weather_filename"],
                                    aimpoints=aimpoints,
                                    soiling=[1.0]*N,
                                    aimpoint_method='Image size priority')

flux_clean = np.array(sp_flux.full_field_flux)

# %% Soiled
outputs_dirty = sp_flux.run_sp_case(case_name=case_name,
                                sp_aimpoint_heur = False, 
                                saveCSV = True,
                                hour_id = settings["hour_idx"], 
                                weather_data = filenames["weather_filename"],
                                aimpoints=aimpoints,
                                soiling=soiling,
                                aimpoint_method='Image size priority')

flux_dirty = np.array(sp_flux.full_field_flux)

 # %% Comparision
width,height = float(sp_flux.receiver_data['length']), float(sp_flux.receiver_data['height'])
tower_height = float(sp_flux.receiver_data['tow_height'])   
extent = (-width/2, width/2, -height/2, height/2)
# Flux maps
fig,ax = plt.subplots(ncols=2,figsize=(16,6.75))
im1 = ax[0].imshow(flux_clean,aspect = 'auto', extent = extent)
im2 = ax[1].imshow(flux_dirty,aspect = 'auto', extent = extent)

ax[0].set_xlabel('horizontal position (m)')
ax[1].set_xlabel('horizontal position (m)')
ax[0].set_ylabel('vertical position (m)')
ax[0].set_title('Clean Flux Map')
ax[1].set_title('Soiled Flux Map')

cbar = fig.colorbar(im1, ax=ax, orientation='vertical')
cbar.set_label('Flux (kW/m²)')
plt.show()
# %% Error maps
# Error maps
exclude = 10
fig2,ax2 = plt.subplots(ncols=2,figsize=(16,6.75))
abs_err = flux_dirty-flux_clean
rel_err = (flux_dirty- flux_clean)/np.abs(flux_clean)*100
rel_err[flux_clean<exclude] = np.nan  # Avoid division by small

im3 = ax2[0].imshow(abs_err,aspect = 'auto', 
                    extent = extent,
                    vmin=None,vmax=None)

im4 = ax2[1].imshow(rel_err,aspect = 'auto', 
                    extent = extent,
                    vmin=None,vmax=None)

ax2[0].set_xlabel('horizontal position (m)')
ax2[0].set_ylabel('vertical position (m)')
ax2[0].set_title('$\Delta$ = |Dirty-Clean|\n')
cbar2 = fig2.colorbar(im3, orientation='vertical')
cbar2.set_label('Flux (kW/m²)')

ax2[1].set_xlabel('horizontal position (m)')
ax2[1].set_title(r'Relative change = $\frac{\Delta}{Clean}$' + f"\n (Clean Flux<{exclude} $kW/m^2$ excluded)")
cbar3 = fig2.colorbar(im4, orientation='vertical')
cbar3.set_label('Error (%)')
plt.show()
