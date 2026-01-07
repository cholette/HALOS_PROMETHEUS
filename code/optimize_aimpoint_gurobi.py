from gurobipy import GRB, quicksum, Model
import numpy as np
from tqdm import tqdm

tol = 1e-6

class gurobi_model:
    def __init__(self,flux_model,all_maps=None,soiling=None,
                 flux_constraint='measurements',
                 params={'max_flux':500,'max_dflux':500}):
        
        flux_constraint = flux_constraint.lower()
        assert flux_constraint in ['aimpoints','measurements'], "flux_constraint must be either ''aimpoints'' or ''measurements'' "
        
        self.params = params
        self.optimized = False
        self.possible_aimpoints = flux_model.receiver.aimpoints
        self.tower_height = flux_model.receiver.tow_height

        N = flux_model.field.GetCoords().shape[0] # number of heliostats
        if all_maps is None:
            # shifted flux maps 
            all_maps = []
            for h in tqdm(range(N),desc='Calculating shifted flux maps'): 
                all_maps += [flux_model.ShiftImage_GenerateSingleHeliostatFluxMaps(h)]

        # N = len(all_maps)# number of heliostats
        M = len(all_maps[0][0]) # number of measurement points (must be a perfect square)
        N_aim = len(all_maps[0]) # number of aimpoints

        self.num_measurements = M
        self.num_heliostats = N
        self.num_aimpoints = N_aim

        if soiling is None:
            print('No Soiling.')
            soiling = [1.0]*N
        
        print('Building flux matrices')
        C = np.zeros((N,N_aim,M))
        for hh,_ in enumerate(all_maps):
            for aa,_ in enumerate(all_maps[hh]):
                C[hh,aa,:] = all_maps[hh][aa]*soiling[hh]
        flux_model.receiver.getSurfaceArea()
        S = flux_model.receiver.surface_area

        print('Instantiating model')
        model = Model('Aimpoint Optimization')

        print('Creating variables')
        y = model.addVars(range(N), range(N_aim),vtype=GRB.BINARY, name='y')
        x = model.addVars(range(M),lb=0,vtype=GRB.CONTINUOUS, name='x')

        print('Adding objective')
        model.setObjective(quicksum(S[m]*x[m] for m in range(M)),GRB.MAXIMIZE)

        print(f'Adding flux constraints ({flux_constraint})')
        if flux_constraint == 'measurements':
            model.addConstrs((x[m] == quicksum((C[h,a,m]*y[h,a] for h in range(N) 
                                                for a in range(N_aim))) for m in range(M)),name='Flux Calculation')
            model.addConstrs((x[m]<=params['max_flux'] for m in range(M)),name='Flux Constraint')
        
        elif flux_constraint == 'aimpoints':
            # find the nearest measurement points to each aim point
            measurement_points = flux_model.receiver.coords
            aim_points = flux_model.receiver.aimpoints
            M_aim = []
            for a in range(N_aim):
                Δ = aim_points[a,:] - measurement_points
                d2 = np.sum(Δ**2,axis=1)
                idx = np.argmin(d2)
                M_aim.append(idx)

            model.addConstrs((x[m] == quicksum((C[h,a,m]*y[h,a] for h in range(N) 
                                                for a in range(N_aim))) for m in range(M)),name='Flux Calculation')
            model.addConstrs((x[m]<=params['max_flux'] for m in M_aim),name='Flux Constraint')

        # print('Adding flux slope constraints')
        # model.addConstrs((x[m]-x[n]<=params['max_dflux'] for m in range(M)
        #                   for n in range(M)),name='ΔFlux Constraint')

        print('Adding selection constraints')
        model.addConstrs( (quicksum((y[h,a] for a in range(N_aim)))<=1 for h in range(N)),name='one_aimpoint')
        self.model = model

    def optimize(self,MIPGap:float=0.01,TimeLimit:int=3600):
        self.model.setParam('TimeLimit',TimeLimit)
        self.model.setParam('MIPGap',MIPGap)
        self.model.optimize()
        self.optimized = True # not necessarily optimal

    def get_optimal_aimpoints(self):
        if self.optimized:
            N,N_aim,M = self.num_heliostats,self.num_aimpoints,self.num_measurements
            y = np.array([self.model.getVarByName(f'y[{ii:d},{jj:d}]').X for ii in range(N) for jj in range(N_aim)])
            y = y.reshape((N,N_aim))
            x = np.array([self.model.getVarByName(f'x[{ii:d}]').X for ii in range(M)])
            
            ap = []
            for ii in range(N):
                idx = np.where(y[ii,:]>=0.5)[0]
                if len(idx)==0:
                    ap.append(-1)
                else:
                    ap.append(idx[0])

            opt_aim = []
            for ii in range(N):
                idx = ap[ii]
                if idx==-1:
                    opt_aim.append([])
                else:
                    apx = -self.possible_aimpoints[idx,0]
                    apy = self.possible_aimpoints[idx,1]
                    apz = self.possible_aimpoints[idx,2]
                    apz = 2.0*self.tower_height-apz 
                    opt_aim.append([apx,apy,apz])
            
            return opt_aim,ap
        
        else:
            print('Call optimize first!')
