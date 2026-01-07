import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def flux_centroid(flux,h,w):
    y, x = np.indices(flux.shape)
    total_intensity = flux.sum()
    x_centroid = (x * flux).sum() / total_intensity
    y_centroid = (y * flux).sum() / total_intensity
    x_centroid = (x_centroid - x.mean())/(x.max() - x.min())
    y_centroid = (y_centroid - y.mean())/(y.max() - y.min())
    return w*x_centroid, h*y_centroid

def inspect_grid(halos,fluxes=None,aimpoint=None,ax=None):
    # Inspect aimpoints and measurement points

    tower_height = halos.receiver.tow_height 
    width = float(halos.receiver.params['length']),
    height = float(halos.receiver.params['height'])

    fig,ax = plt.subplots()
    ax.scatter(halos.receiver.x,
            halos.receiver.z-tower_height,
            s=0.1,label='Measurements')
    ax.scatter(halos.receiver.aim_x,
            halos.receiver.aim_z-tower_height,
            c='r',marker='*',label='Aimpoints')
    
    if fluxes is not None:
        xh,zh = flux_centroid(fluxes['halos'],height,width)
        ax.scatter(xh,-zh, c='cyan',marker='X',s=100,label="Flux Centroid")
    
    if aimpoint is not None:
        ax.scatter(aimpoint[0],tower_height-aimpoint[2],
                   c='magenta',marker='o',s=10,label="Aim point")
    ax.legend()

    # % grid properties
    halos.receiver.getCoords()
    halos.receiver.getAimpoints()
    d = cdist(halos.receiver.aimpoints,halos.receiver.coords)
    d = d.min(axis=1)
    print(f'Aimpoint dist to measurement point: {d.mean():.2e} (mean)')
    print(f'\t\t\t\t    {d.std():.2e} (std)')
    print(f'\t\t\t\t    {d.max():.2e} (max)')

    return fig,ax

def compare_flux_maps(flux,benchmark,surface_area=None):

    if surface_area is not None:
        pw_hal,pw_sp = np.sum(flux*surface_area),np.sum(benchmark*surface_area)
        print(f'Total Power: {pw_hal:.2f} (Model), {pw_sp:.2f} (Benchmark)')

    pk_hal,pk_sp = flux.max(),benchmark.max()
    print(f'Peak fluxes: {pk_hal:.0f} (Model), {pk_sp:.0f} (Benchmark)')

    rmse_sp = np.sqrt(np.mean( (flux-benchmark)**2 ))
    print(f'Model vs Benchmark RMSE: {rmse_sp:.2f}')

    mad_sp = np.mean( np.abs(flux-benchmark) )
    print(f'Model vs Benchmark MAD: {mad_sp:.2f}')

    max_sp = np.max( np.abs(flux-benchmark) )
    print(f'Model vs Benchmark MAX: {max_sp:.2f}')

    return [rmse_sp,mad_sp,max_sp,pw_hal,pw_sp,pk_hal,pk_sp]

def update_pts(file,pts_per_dim, aim_rows, aim_cols,
               aim_h_margin, aim_v_margin):
    
    with open(file, "r") as f:
        lines = f.readlines()

    with open(file, "w") as f:
        for line in lines:
            if "pts_per_dim" in line:
                f.write(f"pts_per_dim,{pts_per_dim:d}\n")
            elif "aim_h_margin" in line:
                f.write(f"aim_h_margin,{aim_h_margin:d}\n")
            elif "aim_v_margin" in line:
                f.write(f"aim_v_margin,{aim_v_margin:d}\n")
            elif "aim_rows" in line:
                f.write(f"aim_rows,{aim_rows:d}\n")
            elif "aim_cols" in line:
                f.write(f"aim_cols,{aim_cols:d}\n")
            else:
                f.write(line)