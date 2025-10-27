# %% Flux Case Studies 2025
""" Flux testing script for case studies as a part of the 2025 Q2 milesone
    for PROMETHEUS project. """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def centroid(flux):
    y, x = np.indices(flux.shape)
    total_intensity = flux.sum()
    x_centroid = (x * flux).sum() / total_intensity
    y_centroid = (y * flux).sum() / total_intensity
    x_centroid = (x_centroid - x.mean())/(x.max() - x.min())
    y_centroid = (y_centroid - y.mean())/(y.max() - y.min())
    return x_centroid, y_centroid

def flux_models(case_filename):
    import inputs 
    import sp_module

    filenames = inputs.readCaseFile(case_filename)
    settings = inputs.readSettingsFile(filenames["settings"])
    N = pd.read_csv(filenames['field_filename']).shape[0]       

    # getting flux model from files
    halos = inputs.getFullFluxModelFromFiles(case_filename,settings['hour_idx'])
    
    # shifted flux maps 
    all_maps = []
    for h in tqdm(range(N)): 
        all_maps += [halos.ShiftImage_GenerateSingleHeliostatFluxMaps(h)]

    # SolarPilot (Hermite)   
    solarpilot_flux = sp_module.SP_Flux(filenames,use_sp_field=False) # solar pilot
    

    # SolarPilot (SolTrace)
    st = sp_module.SP_Flux(filenames,use_sp_field=False) # soltrace
    
    
    models = {'solarpilot':solarpilot_flux,'soltrace':st,'halos':halos}
    params = {'settings':settings,'filenames':filenames}
    return models,all_maps,params

def get_fluxes(models,all_maps,params,aimpoints,soiling):

    settings = params['settings']
    filenames = params['filenames']
    central_fluxes = models['halos'].parallel_flux_maps
    meas_grid_shape = central_fluxes[0].shape
    flux_halos = np.zeros(meas_grid_shape)
    for ii in range(N):
        flux_halos += all_maps[ii][aimpoint_idx[ii]].reshape(meas_grid_shape)*soiling[ii]

    outputs_sp = models['solarpilot'].run_sp_case(   case_name=case_name,
                                                sp_aimpoint_heur = False, 
                                                saveCSV = True,
                                                hour_id = settings["hour_idx"], 
                                                weather_data = filenames["weather_filename"],
                                                aimpoints=aimpoints,
                                                soiling=soiling,
                                                aimpoint_method=None,
                                                use_soltrace=False)
    
    flux_sp = np.array(models['solarpilot'].full_field_flux)

    outputs_st = models['soltrace'].run_sp_case(  case_name=case_name,
                                    sp_aimpoint_heur = False, 
                                    saveCSV = True,
                                    hour_id = settings["hour_idx"], 
                                    weather_data = filenames["weather_filename"],
                                    aimpoints=aimpoints,
                                    soiling=soiling,
                                    aimpoint_method=None,
                                    use_soltrace=use_soltrace,
                                    max_rays = 100000000,
                                    min_rays = 1000000)
    
    flux_st = np.array(models['soltrace'].full_field_flux)

    fluxes = {'solarpilot':flux_sp,'soltrace':flux_st,'halos':flux_halos}

    return fluxes

def compare_sp_and_halos(models,fluxes):
     
    flux_st = fluxes['soltrace']
    flux_sp = fluxes['solarpilot']
    flux_halos = fluxes['halos']
    
    st = models['soltrace']
    sp = models['solarpilot']
    halos = models['halos']


    width,height = float(sp.receiver_data['length']), float(sp.receiver_data['height'])
     
       
    # Comparision plots
    stc = centroid(flux_st)
    spc = centroid(flux_sp)
    halc = centroid(flux_halos)

    extent = (-width/2, width/2, -height/2, height/2)
    # Flux maps
    fig,ax = plt.subplots(ncols=3,figsize=fsz)

    im1 = ax[0].imshow(flux_sp,aspect = 'auto', extent = extent)
    ax[0].scatter(spc[0]*width, -spc[1]*height, color='red', label='SolarPilot Centroid ', marker='x')
    ax[0].set_xlabel('horizontal position (m)')
    ax[0].set_title('SolarPilot Flux Map (Hermite)')
    
    
    im2 = ax[1].imshow(flux_st,aspect = 'auto', extent = extent)
    ax[1].scatter(stc[0]*width, -stc[1]*height, color='red', label='SolTrace Centroid', marker='x')
    ax[1].set_xlabel('horizontal position (m)')
    ax[1].set_title('SolarPilot Flux Map (SolTrace)')

    im3 = ax[2].imshow(flux_halos,aspect = 'auto', extent = extent)
    ax[2].scatter(halc[0]*width, -halc[1]*height, color='blue', label='HALOS Centroid', marker='x')
    ax[2].set_xlabel('horizontal position (m)')        
    ax[2].set_title('HALOS Flux Map')

    cbar = fig.colorbar(im1, ax=ax, orientation='vertical')
    cbar.set_label('Flux (kW/m²)')
    plt.show()

    # Error maps
    fig2,ax2 = plt.subplots(ncols=3,figsize=fsz)
    abs_err_sp = flux_halos-flux_sp
    abs_err_st = flux_halos-flux_st
    # rel_err = np.abs(flux_halos-flux_sp)/np.abs(flux_sp)*100
    # rel_err[flux_sp<exclude] = np.nan  # Avoid division by zero
    
    # get value range to ensure colormap is the same
    vmin = min(abs_err_sp.min(),abs_err_st.min())
    vmax = max(abs_err_sp.max(),abs_err_st.max())

    im4 = ax2[0].imshow(abs_err_sp,aspect = 'auto', 
                        extent = extent,
                        vmin=vmin,vmax=vmax)

    im5 = ax2[1].imshow(abs_err_st,aspect = 'auto', 
                        extent = extent,
                        vmin=vmin,vmax=vmax)
    
    im6 = ax2[2].imshow(flux_sp-flux_st,aspect = 'auto', 
                        extent = extent,
                        vmin=vmin,vmax=vmax)
    
    ax2[0].set_xlabel('horizontal position (m)')
    ax2[0].set_ylabel('vertical position (m)')
    ax2[0].set_title('error = HALOS -SolarPilot\n')
    

    ax2[1].set_xlabel('horizontal position (m)')
    ax2[1].set_title('error = HALOS -SolTrace \n')

    ax2[2].set_xlabel('horizontal position (m)')
    ax2[2].set_title('error = SolarPILOT - SolTrace \n')
    

    cbar3 = fig.colorbar(im4, ax=ax2, orientation='vertical')
    cbar3.set_label('Flux Error (kW/m²)')
    plt.show()    

def optimize(models,all_maps):
    from optimize_aimpoint_gurobi import gurobi_model
    gm = gurobi_model(models['halos'],all_maps)
    gm.model.setParam('MIPGap', 0.01)
    gm.model.optimize()

    N = len(all_maps)# number of heliostats
    N_aim = len(all_maps[0]) # number of aimpoints
    opt_aim = []
    for ii in range(N):
        jj = 0
        while (jj < N_aim) and (len(opt_aim)<ii):
            if gm.model.getVarByName(f'y[{ii},{jj}]'):
                opt_aim.append(jj)

    return gm

# %% Main    
if __name__ == "__main__":

    case_filename = "./../case_inputs/flat_50_ca_case.csv"
    case_name = "flat_daggett_50"
    use_soltrace = False
    simulate_soiling = False
    fsz = (16,6.75)

    models,all_maps,params = flux_models(case_filename)

    # Set aimpoints for both HALOS and SolarPILOT
    aimpoint_idx = []
    aimpoints = []
    possible_aimpoints = models['halos'].receiver.aimpoints
    tower_height = float(models['solarpilot'].receiver_data['tow_height'])  
    N = models['halos'].field.GetCoords().shape[0]
    N_aim = possible_aimpoints.shape[0]
    for i in range(N):
        # idx = N_aim // 2  # Use the center aimpoint
        idx = np.random.randint(0,N_aim)

        if np.random.rand() < 0.5:
            idx = 8
        else:
            idx = 40

        aimpoint_idx.append(idx)
        apx = -possible_aimpoints[idx,0]
        apy = possible_aimpoints[idx,1]
        apz = possible_aimpoints[idx,2]
        apz = 2.0*tower_height-apz 
        aimpoints.append([apx,apy,apz])
    
    # Set soiling factor (equivalent to instantaneous cleanliness, 1.0 = perfectly clean)
    soiling = [1.0]*N
    if simulate_soiling:
        for ii in range(200,500):
            soiling[ii] = 0.75  # Example soiling condition for the first heliostat
 
    fluxes = get_fluxes(models,all_maps,params,aimpoints,soiling)
    compare_sp_and_halos(models,fluxes)

        
    pk_hal,pk_sp, pk_st = fluxes['halos'].max(),fluxes['solarpilot'].max(),fluxes['soltrace'].max()
    print(f'Peak fluxes: {pk_hal:.0f} (HALOS), {pk_sp:.0f} (SolarPILOT),{pk_st:.0f} (SolTrace)')

    pts = int(models['solarpilot'].receiver_data['pts_per_dim'])
    S = models['halos'].receiver.surface_area.reshape((pts,pts))
    pw_hal,pw_sp,pw_st = np.sum(fluxes['halos']*S),np.sum(fluxes['solarpilot']*S),np.sum(fluxes['soltrace']*S)
    print(f'Total Power: {pw_hal:.2f} (HALOS), {pw_sp:.2f} (SolarPILOT), {pw_st:.2f} (SolTrace)')

# %%

