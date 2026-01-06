# %% SolarPILOT experiments
import inputs
import sp_module
import numpy as np
from mec_comparison_utilities import inspect_grid, compare_flux_maps, update_pts
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

case_filename = "./../case_inputs/flat_50_ca_case.csv"
case_name = "flat_daggett_50"
filenames = inputs.readCaseFile(case_filename)
settings = inputs.readSettingsFile(filenames["settings"])

desired_aimpoints = [[7.0,0.0,4.0],[-7.0,0.0,-4.0],[0.0,0.0,4.0],[0.0,0.0,-4.0],
                     [7.0,0.0,-4.0],[-7.0,0.0,4.0]]
# desired_aimpoints = [[7.0,0.0,2.0],[-7.0,0.0,-2.0]]
probs = [1/len(desired_aimpoints)]*len(desired_aimpoints)
save_aimpoint = True # Only makes sense for a single aimpoint
np_seed = 42

pts = [5,7,10,13,15,19,20,25,30,31,35,37,40,43,45,49,50,73,75,90,91,100,103]
# pts = [13,15]
aim_h_margin,aim_v_margin = 0,0

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
        solarpilot_flux = sp_module.SP_Flux(filenames,use_sp_field=False) # solar pilot
        N = solarpilot_flux.num_heliostats
        tow_height = float(solarpilot_flux.receiver_data['tow_height'])

        # set aimopoint to be the one that's closest to the desired aimpoint
        # da = desired_aimpoint + np.array([0,0,tow_height])
        aimpoint_grid = halos.receiver.aimpoints
        desired_idxs = []
        desired_aim = []
        for ii,da in enumerate(desired_aimpoints):
            da_temp = da.copy()
            da_temp[-1] += tow_height
            d = np.sqrt(np.sum( (aimpoint_grid-da_temp)**2, axis=1))
            idx = np.where(d==min(d))[0][0]
            desired_aim.append(aimpoint_grid[idx,:])
            desired_idxs.append(idx)
            print(f'Adjusted desired aimpoint: [{desired_aim[ii][0]:.2f},{desired_aim[ii][1]:.2f},{desired_aim[ii][2]:.2f}] ({min(d):.3f} away from desired)')
        
        aimpoint = []
        aimpoint_idx = []
        rng = np.random.default_rng(seed=np_seed)
        for ii in range(N):
            idx = rng.choice(len(desired_aim),p=probs)
            idx = desired_idxs[idx]
            aimpoint_idx.append(idx)
            aimpoint.append(aimpoint_grid[idx,:])
        

        # get aimpoint index to halos aimpoint grid
        aimpoint_halos = aimpoint.copy()
        # aimpoint_halos[-1] += tow_height
        
        # shifted flux maps 
        meas_grid_shape = halos.parallel_flux_maps[0].shape
        all_maps = []
        N = halos.field.coords.shape[0]
        for h in range(N): 
            all_maps += [halos.ShiftImage_GenerateSingleHeliostatFluxMaps(h)]


        flux_halos = np.zeros_like(halos.parallel_flux_maps[0])
        # for h in range(len(halos.parallel_flux_maps)):
        #     flux_halos += halos.parallel_flux_maps[h]
        for ii in range(N):
            flux_halos += all_maps[ii][aimpoint_idx[ii]].reshape(meas_grid_shape)


        # SolarPilot (Hermite)   
        aimpoint_sp = aimpoint.copy()
        # aimpoint_sp[-1] += tow_height
        # aimpoints = np.array(aimpoint) + np.array([0,0,tow_height])
        # aimpoint_sp = aimpoint_sp * np.ones((N,1))

        outputs_sp = solarpilot_flux.run_sp_case(   case_name=case_name,
                                                        sp_aimpoint_heur = False, 
                                                        saveCSV = True,
                                                        hour_id = settings["hour_idx"], 
                                                        weather_data = filenames["weather_filename"],
                                                        aimpoints=aimpoint_sp,
                                                        soiling=None,
                                                        aimpoint_method=None,
                                                        use_soltrace=False)
            
        # % error
        plt.show() # to ensure that new plot shows up on a new figure
        err = flux_halos-solarpilot_flux.full_field_flux
        solarpilot_flux.plot_flux_map(err)

        inspect_grid(halos)
        pts = halos.receiver.params['pts_per_dim']
        sp_flux = np.array(solarpilot_flux.full_field_flux)
        S = halos.receiver.surface_area.reshape((pts,pts))
        res = compare_flux_maps(flux_halos,sp_flux,surface_area=S)

        # put into results dict
        res.insert(0,pt)

        if save_aimpoint:
            res.append(aimpoint[0])
            res.append(aimpoint[1])
            res.append(aimpoint[2])
        else:
            res.append(np.nan)
            res.append(np.nan)
            res.append(np.nan)
        for ii,k in enumerate(results.keys()):
            results[k].append(res[ii])

    results['aim_h_margin'] = aim_h_margin
    results['aim_v_margin'] = aim_v_margin
    df = pd.DataFrame(results)
# %%
