# %% Flux Case Studies 2025
""" Flux testing script for case studies as a part of the 2025 Q2 milesone
    for PROMETHEUS project. """

import inputs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sp_module

def centroid(flux):
    y, x = np.indices(flux.shape)
    total_intensity = flux.sum()
    x_centroid = (x * flux).sum() / total_intensity
    y_centroid = (y * flux).sum() / total_intensity
    x_centroid = (x_centroid - x.mean())/(x.max() - x.min())
    y_centroid = (y_centroid - y.mean())/(y.max() - y.min())
    return x_centroid, y_centroid

    
case_filename = "./../case_inputs/flat_50_ca_case.csv"
case_name = "flat_daggett_50"
filenames = inputs.readCaseFile(case_filename)
settings = inputs.readSettingsFile(filenames["settings"])
N = pd.read_csv(filenames['field_filename']).shape[0]
tower_height = float(pd.read_csv(filenames['receiver_filename'],index_col=0,names=['name','value']).loc['tow_height'].value)
fsz = (16,6.75)

if __name__ == "__main__":
    # Set soiling factor (equivalent to instantaneous cleanliness, 1.0 = perfectly clean)
    soiling = [1.0]*N
    for ii in range(200,500):
        soiling[ii] = 0.75  # Example soiling condition for the first heliostat

     # HALOS
    halos_flux_model = inputs.getFullFluxModelFromFiles(case_filename,settings['hour_idx'])
    central_fluxes = halos_flux_model.parallel_flux_maps
    meas_grid_shape = central_fluxes[0].shape

    ## Central flux from HALOS
    # flux_halos = np.zeros(central_fluxes[0].shape)
    # for key,item in central_fluxes.items():
    #     flux_halos += item

    # shifted flux maps  
    all_maps = [halos_flux_model.ShiftImage_GenerateSingleHeliostatFluxMaps(h) for h in range(N)]

    # Set aimpoints for both HALOS and SolarPILOT
    aimpoint_idx = []
    aimpoints = []
    possible_aimpoints = halos_flux_model.receiver.aimpoints
    N_aim = possible_aimpoints.shape[0]
    for i in range(N):
        idx = N_aim // 2  # Use the center aimpoint
        # idx = np.random.randint(0,N_aim)
        # if np.random.rand() < 0.5:
        #     idx = 8
        # else:
        #     idx = 40

        aimpoint_idx.append(idx)
        apx = -possible_aimpoints[idx,0]
        apy = possible_aimpoints[idx,1]
        apz = possible_aimpoints[idx,2]
        apz = 2.0*tower_height-apz 
        aimpoints.append([apx,apy,apz])
    
    flux_halos = np.zeros(meas_grid_shape)
    for ii in range(N):
        flux_halos += all_maps[ii][aimpoint_idx[ii]].reshape(meas_grid_shape)*soiling[ii]
    
    # %% SolarPilot
    sp_flux = sp_module.SP_Flux(filenames,use_sp_field=False)
    outputs = sp_flux.run_sp_case(  case_name=case_name,
                                    sp_aimpoint_heur = False, 
                                    saveCSV = True,
                                    hour_id = settings["hour_idx"], 
                                    weather_data = filenames["weather_filename"],
                                    aimpoints=aimpoints,
                                    soiling=soiling,
                                    aimpoint_method=None)
    
    flux_sp = np.array(sp_flux.full_field_flux)
    width,height = float(sp_flux.receiver_data['length']), float(sp_flux.receiver_data['height'])
    tower_height = float(sp_flux.receiver_data['tow_height'])   

   
    # %% Comparision plots
    spc = centroid(flux_sp)
    halc = centroid(flux_halos)

    extent = (-width/2, width/2, -height/2, height/2)
    # Flux maps
    fig,ax = plt.subplots(ncols=2,figsize=fsz)
    im1 = ax[0].imshow(flux_sp,aspect = 'auto', extent = extent)
    ax[0].scatter(spc[0]*width, -spc[1]*height, color='red', label='SolarPilot Centroid', marker='x')
    im2 = ax[1].imshow(flux_halos,aspect = 'auto', extent = extent)
    ax[1].scatter(halc[0]*width, -halc[1]*height, color='blue', label='HALOS Centroid', marker='x')
    
    ax[0].set_xlabel('horizontal position (m)')
    ax[1].set_xlabel('horizontal position (m)')
    ax[0].set_ylabel('vertical position (m)')
    ax[0].set_title('SolarPilot Flux Map')
    ax[1].set_title('HALOS Flux Map')

    cbar = fig.colorbar(im1, ax=ax, orientation='vertical')
    cbar.set_label('Flux (kW/m²)')
    plt.show()

    # Error maps
    exclude = 10
    fig2,ax2 = plt.subplots(ncols=2,figsize=fsz)
    abs_err = np.abs(flux_halos-flux_sp)
    rel_err = np.abs(flux_halos-flux_sp)/np.abs(flux_sp)*100
    rel_err[flux_sp<exclude] = np.nan  # Avoid division by zero
    
    im3 = ax2[0].imshow(abs_err,aspect = 'auto', 
                        extent = extent,
                        vmin=0,vmax=None)

    im4 = ax2[1].imshow(rel_err,aspect = 'auto', 
                        extent = extent,
                        vmin=0,vmax=25)
    
    ax2[0].set_xlabel('horizontal position (m)')
    ax2[0].set_ylabel('vertical position (m)')
    ax2[0].set_title('Absolute Error = |HALOS -SolarPilot|\n')
    cbar2 = fig2.colorbar(im3, orientation='vertical')
    cbar2.set_label('Flux (kW/m²)')

    ax2[1].set_xlabel('horizontal position (m)')
    ax2[1].set_title(f'Relative Error = |HALOS-SolarPilot|/|SolarPilot|\n(Flux<{exclude} kW/m^2 excluded)')
    cbar3 = fig2.colorbar(im4, orientation='vertical')
    cbar3.set_label('Error (%)')
    plt.show()
    

