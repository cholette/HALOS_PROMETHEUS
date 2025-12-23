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
pts = [21]
aim_h_margin,aim_v_margin = 1,1
MIPGap = 0.01

results = {'pts_per_dim':[],
           'rmse':[],
           'mad': [],
           'max_abs_dev': [],
           'total_power_halos':[],
           'total_power_solarpilot':[],
           'peak_flux_halos':[],
           'peak_flux_solarpilot':[],
           'actual_aimpoint_x':[],
           'actual_aimpoint_y':[],
           'actual_aimpoint_z':[]}

# %% HALOS
if __name__ == "__main__":

    for pt in tqdm(pts):

        update_pts(filenames['receiver_filename'],pt,aim_h_margin,aim_h_margin)

        # getting flux model from files
        halos = inputs.getFullFluxModelFromFiles(case_filename,settings['hour_idx'])

        update_pts(filenames['receiver_filename'],10*pt,aim_h_margin,aim_h_margin)
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
        aim = gurobi_model(halos,all_maps,flux_constraint='measurements')
        build_time = time.perf_counter() - start
        print(f'...Done! {build_time:.2f} seconds')

        start = time.perf_counter()
        aim.optimize(MIPGap=MIPGap)
        optimization_time = time.perf_counter() - start
        print(f'...Done! {optimization_time:.2f} seconds')
        print(f'Build + Optimize time {build_time + optimization_time:.2f} seconds')
                
        opt_aim_coords,opt_aim_idxs = aim.get_optimal_aimpoints()
        flux_halos = np.zeros_like(halos.parallel_flux_maps[0])
        for ii in range(N):
            if opt_aim_idxs[ii]==-1:
                print(f'Heliostat {ii} is defocused')
            else:
                flux_halos += all_maps[ii][opt_aim_idxs[ii]].reshape(meas_grid_shape)
        
        
        outputs_sp = solarpilot_flux.run_sp_case(   case_name=case_name,
                                                        sp_aimpoint_heur = False, 
                                                        saveCSV = True,
                                                        hour_id = settings["hour_idx"], 
                                                        weather_data = filenames["weather_filename"],
                                                        aimpoints=opt_aim_coords,
                                                        soiling=None,
                                                        aimpoint_method=None,
                                                        use_soltrace=False)
            
        # % error
        plt.show() # to ensure that new plot shows up on a new figure
        # err = flux_halos-solarpilot_flux.full_field_flux
        # solarpilot_flux.plot_flux_map(err)
        flux_sp = np.array(solarpilot_flux.full_field_flux)
        S = halos.receiver.surface_area.reshape((pt,pt))
        power_sp = outputs_sp['power']
        print(f'HALOS Power: {np.sum(flux_halos*S):.2f} W, Peak Flux {flux_halos.max():.2f} W/m2')
        print(f'SolarPILOT Power (100x grid): {power_sp:.2f} W, Peak Flux: {flux_sp.max():.2f} W/m2')

        inspect_grid(halos)
    #     pts = halos.receiver.params['pts_per_dim']
    #     sp_flux = np.array(solarpilot_flux.full_field_flux)
    #     S = halos.receiver.surface_area.reshape((pts,pts))
    #     res = compare_flux_maps(flux_halos,sp_flux,surface_area=S)

    #     # put into results dict
    #     res.insert(0,pt)

    #     if save_aimpoint:
    #         res.append(aimpoint[0])
    #         res.append(aimpoint[1])
    #         res.append(aimpoint[2])
    #     else:
    #         res.append(np.nan)
    #         res.append(np.nan)
    #         res.append(np.nan)
    #     for ii,k in enumerate(results.keys()):
    #         results[k].append(res[ii])

    # results['aim_h_margin'] = aim_h_margin
    # results['aim_v_margin'] = aim_v_margin
    # df = pd.DataFrame(results)
# %%
