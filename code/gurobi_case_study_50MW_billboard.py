# %% SolarPILOT experiments
import inputs
import sp_module
import numpy as np
from mec_comparison_utilities import inspect_grid, compare_flux_maps, update_pts
import matplotlib.pyplot as plt
import pandas as pd
from optimize_aimpoint_gurobi import gurobi_model
import time

case_filename = "./../case_inputs/flat_50_ca_case.csv"
case_name = "flat_daggett_50"
filenames = inputs.readCaseFile(case_filename)
settings = inputs.readSettingsFile(filenames["settings"])
pts = [33]
aim_rows,aim_cols=7,7
aim_h_margin,aim_v_margin = 1,1
MIPGap = 0.01
flux_constraint_points = 'measurements' # 'aimpoints' or 'measurements'
sp_grid_factor = 10

df_soiling = pd.read_excel('../flat_dagget_50_soiling/daggett_field_soiled.xlsx',sheet_name='second_surface')
soil_col = 'soiling_factor @ 2018-08-03 09:00'


# %% build optimization model and plot
def optimize_scenario(halos,all_maps,soiling,optimize_clean=True):
    start = time.perf_counter()

    if optimize_clean:
        print('Constructing clean field optimization model...')
        aim = gurobi_model(halos,all_maps,
                            flux_constraint=flux_constraint_points,
                            soiling=None)
    else:
        print('Constructing soilied field optimization model...')
        aim = gurobi_model(halos,all_maps,
                            flux_constraint=flux_constraint_points,
                            soiling=soiling)
        
    opt_build_time = time.perf_counter() - start
    print(f'...Done! Optimization model build in {opt_build_time:.2f} seconds')

    start = time.perf_counter()
    aim.optimize(MIPGap=MIPGap)
    opt_time = time.perf_counter() - start
    print(f'...Done! Optimized in {opt_time:.2f} seconds')
    print(f'Optimization build + Optimize time {opt_build_time + opt_time:.2f} seconds')
            
    opt_aim_coords,opt_aim_idxs = aim.get_optimal_aimpoints()
    flux_halos = np.zeros_like(halos.parallel_flux_maps[0])
    for ii in range(N):
        if opt_aim_idxs[ii]==-1:
            print(f'Heliostat {ii} is defocused')
        else:
            flux_halos += all_maps[ii][opt_aim_idxs[ii]].reshape(meas_grid_shape)*soiling[ii]

    return flux_halos,opt_aim_coords

# %% HALOS
if __name__ == "__main__":

    for pt in pts:

        update_pts(filenames['receiver_filename'],pt,
                   aim_rows,aim_cols,aim_h_margin,aim_h_margin)

        # getting flux model from files
        start = time.perf_counter()
        halos = inputs.getFullFluxModelFromFiles(case_filename,settings['hour_idx'])
        flux_build_time = time.perf_counter() - start
        print(f'...Done! HALOS flux model built in: {flux_build_time:.2f} seconds')

        print(f'Setting Soiling factors...')
        soiling = []
        for ii in range(halos.field.coords.shape[0]):
            x,y,z =  halos.field.coords[ii]
            x_match = np.isclose(df_soiling['Loc. X'][ii],x)
            y_match = np.isclose(df_soiling['Loc. Y'][ii],y)
            z_match = np.isclose(df_soiling['Loc. Z'][ii],z)
            if x_match and y_match and z_match:
                soiling += [float(df_soiling[soil_col][ii])]
            else:
                raise ValueError(f'No match found for heliostat {ii} in soiling dataframe')

        update_pts(filenames['receiver_filename'],sp_grid_factor*pt,
                   aim_rows,aim_cols,aim_h_margin,aim_h_margin)
        
        solarpilot_flux = sp_module.SP_Flux(filenames,use_sp_field=False) # solar pilot
        
        
        N = solarpilot_flux.num_heliostats
        tow_height = float(solarpilot_flux.receiver_data['tow_height'])

        # shifted flux maps 
        meas_grid_shape = halos.parallel_flux_maps[0].shape
        all_maps = []
        N = halos.field.coords.shape[0]
        for h in range(N): 
            all_maps += [halos.ShiftImage_GenerateSingleHeliostatFluxMaps(h)]
        
        # optimize aimpoints
        flux_halos_clean,opt_aim_coords_clean = optimize_scenario(halos,all_maps,soiling=soiling,optimize_clean=True)
        flux_halos_dirty,opt_aim_coords_dirty = optimize_scenario(halos,all_maps,soiling=soiling,optimize_clean=False)
        
        
        start = time.perf_counter()
        outputs_sp_clean = solarpilot_flux.run_sp_case(   case_name=case_name,
                                                    sp_aimpoint_heur = False, 
                                                    saveCSV = True,
                                                    hour_id = settings["hour_idx"], 
                                                    weather_data = filenames["weather_filename"],
                                                    aimpoints=opt_aim_coords_clean,
                                                    soiling=soiling,
                                                    aimpoint_method=None,
                                                    use_soltrace=False)
        
        flux_sp_clean = np.array(solarpilot_flux.full_field_flux)*1.0 # make a copy to avoid overwriting when we run the dirty case
        
        outputs_sp_dirty = solarpilot_flux.run_sp_case(   case_name=case_name,
                                                    sp_aimpoint_heur = False, 
                                                    saveCSV = True,
                                                    hour_id = settings["hour_idx"], 
                                                    weather_data = filenames["weather_filename"],
                                                    aimpoints=opt_aim_coords_dirty,
                                                    soiling=soiling,
                                                    aimpoint_method=None,
                                                    use_soltrace=False)
        
        flux_sp_dirty = np.array(solarpilot_flux.full_field_flux)*1.0 # make a copy to avoid overwriting when we run the dirty case

        sp_time = time.perf_counter() - start
        print(f'...Done! SolarPILOT simulation of optimal strategy took {sp_time:.2f} seconds')

    # %% Output        
    # % error
    plt.show() # to ensure that new plot shows up on a new figure
    # err = flux_halos-solarpilot_flux.full_field_flux
    # solarpilot_flux.plot_flux_map(err)
    S = halos.receiver.surface_area.reshape((pt,pt))
    power_sp_clean = outputs_sp_clean['power']
    power_sp_dirty = outputs_sp_dirty['power']
    print('--------- Clean optimized ---------')
    print(f'HALOS Power: {np.sum(flux_halos_clean*S):.2f} kW, Peak Flux {flux_halos_clean.max():.2f} kW/m2')
    print(f'SolarPILOT Power ({sp_grid_factor:d}x grid): {power_sp_clean:.2f} kW, Peak Flux: {flux_sp_clean.max():.2f} kW/m2')

    plot_height = float(solarpilot_flux.receiver_data["height"])
    plot_width = float(solarpilot_flux.receiver_data["length"])
    pr,pc = np.unravel_index(np.argmax(flux_sp_clean),flux_sp_clean.shape)
    y_peak = (pr/flux_sp_clean.shape[0])*plot_height
    x_peak = (pc/flux_sp_clean.shape[1])*plot_width - plot_width/2.0
    fig,ax = inspect_grid(halos)
    ax.scatter(x_peak,y_peak-plot_height/2.0,marker='*',c='g',label='Peak Flux Position')
    ax.legend()
    ax.set_title('Grid and peak flux position - Clean Optimization')
    
    fig,ax = plt.subplots()
    im = plt.imshow(flux_sp_clean, aspect = 'auto', extent = (-plot_width/2, plot_width/2, 0, plot_height))
    plt.colorbar(im)
    plt.ylabel('Receiver vertical position [m]')
    plt.xlabel('Receiver horizontal position [m]')
    ax.scatter(x_peak,y_peak,marker='*',c='g',label='Peak Flux Position')
    ax.set_title('Flux map - Clean Optimization')
    
    
    print('--------- Dirty optimized ---------')
    print(f'HALOS Power: {np.sum(flux_halos_dirty*S):.2f} kW, Peak Flux {flux_halos_dirty.max():.2f} kW/m2')
    print(f'SolarPILOT Power ({sp_grid_factor:d}x grid): {power_sp_dirty:.2f} kW, Peak Flux: {flux_sp_dirty.max():.2f} kW/m2')
    

    plot_height = float(solarpilot_flux.receiver_data["height"])
    plot_width = float(solarpilot_flux.receiver_data["length"])
    pr,pc = np.unravel_index(np.argmax(flux_sp_dirty),flux_sp_dirty.shape)
    y_peak = (pr/flux_sp_dirty.shape[0])*plot_height
    x_peak = (pc/flux_sp_dirty.shape[1])*plot_width - plot_width/2.0
    fig,ax = inspect_grid(halos)
    ax.scatter(x_peak,y_peak-plot_height/2.0,marker='*',c='g',label='Peak Flux Position')
    ax.legend()
    ax.set_title('Grid and peak flux position - Dirty Optimization')

    fig,ax = plt.subplots()
    im = plt.imshow(flux_sp_dirty, aspect = 'auto', extent = (-plot_width/2, plot_width/2, 0, plot_height))
    plt.colorbar(im)
    plt.ylabel('Receiver vertical position [m]')
    plt.xlabel('Receiver horizontal position [m]')
    ax.scatter(x_peak,y_peak,marker='*',c='g',label='Peak Flux Position')
    ax.set_title('Flux map - Dirty Optimization')
        


# %%
