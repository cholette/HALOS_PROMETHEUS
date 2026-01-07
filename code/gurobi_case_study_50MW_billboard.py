# %% SolarPILOT experiments
import inputs
import sp_module
import numpy as np
from mec_comparison_utilities import inspect_grid, compare_flux_maps, update_pts
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        print('Building model...')
        start = time.perf_counter()
        aim = gurobi_model(halos,all_maps,flux_constraint=flux_constraint_points)
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
                flux_halos += all_maps[ii][opt_aim_idxs[ii]].reshape(meas_grid_shape)
        
        start = time.perf_counter()
        outputs_sp = solarpilot_flux.run_sp_case(   case_name=case_name,
                                                    sp_aimpoint_heur = False, 
                                                    saveCSV = True,
                                                    hour_id = settings["hour_idx"], 
                                                    weather_data = filenames["weather_filename"],
                                                    aimpoints=opt_aim_coords,
                                                    soiling=None,
                                                    aimpoint_method=None,
                                                    use_soltrace=False)
        sp_time = time.perf_counter() - start
        print(f'...Done! SolarPILOT simulation of optimal strategy took {sp_time:.2f} seconds')
            
        # % error
        plt.show() # to ensure that new plot shows up on a new figure
        # err = flux_halos-solarpilot_flux.full_field_flux
        # solarpilot_flux.plot_flux_map(err)
        flux_sp = np.array(solarpilot_flux.full_field_flux)
        S = halos.receiver.surface_area.reshape((pt,pt))
        power_sp = outputs_sp['power']
        print(f'HALOS Power: {np.sum(flux_halos*S):.2f} kW, Peak Flux {flux_halos.max():.2f} kW/m2')
        print(f'SolarPILOT Power ({sp_grid_factor:d}x grid): {power_sp:.2f} kW, Peak Flux: {flux_sp.max():.2f} kW/m2')

        plot_height = float(solarpilot_flux.receiver_data["height"])
        plot_width = float(solarpilot_flux.receiver_data["length"])
        pr,pc = np.unravel_index(np.argmax(flux_sp),flux_sp.shape)
        y_peak = (pr/flux_sp.shape[0])*plot_height
        x_peak = (pc/flux_sp.shape[1])*plot_width - plot_width/2.0

        fig,ax = inspect_grid(halos)
        ax.scatter(x_peak,y_peak-plot_height/2.0,marker='*',c='g',label='Peak Flux Position')
        ax.legend()
        
        
        fig,ax = plt.subplots()
        im = plt.imshow(flux_sp, aspect = 'auto', extent = (-plot_width/2, plot_width/2, 0, plot_height))
        plt.colorbar(im)
        plt.ylabel('Receiver vertical position [m]')
        plt.xlabel('Receiver horizontal position [m]')
        ax.scatter(x_peak,y_peak,marker='*',c='g',label='Peak Flux Position')
        


# %%
