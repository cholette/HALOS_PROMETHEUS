from gurobipy import GRB, quicksum, Model
import numpy as np
from tqdm import tqdm


class gurobi_model:
    def __init__(self,flux_model,all_maps=None,params={'max_flux':500,'max_dflux':500}):
        """ Insert docstring here"""
        self.params = params

        if all_maps is None:
            # shifted flux maps 
            all_maps = []
            for h in tqdm(range(N),desc='Calculating shifted flux maps'): 
                all_maps += [flux_model.ShiftImage_GenerateSingleHeliostatFluxMaps(h)]

        N = len(all_maps)# number of heliostats
        M = len(all_maps[0][0]) # number of measurement points (must be a perfect square)
        N_aim = len(all_maps[0]) # number of aimpoints
        
        print('Building flux matrices')
        C = np.zeros((N,N_aim,M))
        for hh,_ in enumerate(all_maps):
            for aa,_ in enumerate(all_maps[hh]):
                C[hh,aa,:] = all_maps[hh][aa]
        flux_model.receiver.getSurfaceArea()
        S = flux_model.receiver.surface_area

        print('Instantiating model')
        model = Model('Aimpoint Optimization')

        print('Creating variables')
        y = model.addVars(range(N), range(N_aim),vtype=GRB.BINARY, name='aiming')
        x = model.addVars(range(M),lb=0,vtype=GRB.CONTINUOUS, name='flux')

        print('Adding objective')
        model.setObjective(quicksum(S[m]*x[m] for m in range(M)),GRB.MAXIMIZE)

        print('Adding flux constraints')
        model.addConstrs((x[m] == quicksum((C[h,a,m]*y[h,a] for h in range(N) 
                                             for a in range(N_aim))) for m in range(M)),name='Flux Calculation')
        model.addConstrs((x[m]<=params['max_flux'] for m in range(M)),name='Flux Constraint')

        # print('Adding flux slope constraints')
        # model.addConstrs((x[m]-x[n]<=params['max_dflux'] for m in range(M)
        #                   for n in range(M)),name='ΔFlux Constraint')

        print('Adding selection constraints')
        model.addConstrs( (quicksum((y[h,a] for a in range(N_aim)))<=1 for h in range(N)),name='one_aimpoint')
        
        
        self.model = model
