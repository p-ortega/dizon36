# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:48:41 2025

@author: afoster
"""

import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy

cwd = os.getcwd()
dis_ws = os.path.join(cwd, 'dis')
props_ws = os.path.join(cwd, 'props')

model_ws = os.path.join(os.getcwd(), 'model')
if not os.path.exists(model_ws):
    os.makedirs(model_ws)

# define mf6 exectuable path
mf6_exe_ws = os.path.join(cwd, 'bin', 'win', 'mf6.exe')  # Path to the executable
# define the full path for the destination file
mf6_exe_ws_model = os.path.join(model_ws, 'mf6.exe')
# copy the executable file to the new folder
shutil.copy(mf6_exe_ws, mf6_exe_ws_model)

def reformat_arrays(input_file, output_file, rows, cols):
    # Read the content of the file
    with open(input_file, 'r') as f:
        content = f.read()

    # Flatten the values into a 1D array and remove spaces or NaNs
    values = [val for val in content.split() if val.strip() and val.lower() != 'nan']

    # Convert values to a numpy array
    values_array = np.array(values)

    # Ensure the array can be reshaped to the desired shape
    if len(values_array) != rows * cols:
        raise ValueError("The number of values does not match the specified shape")

    # Reshape the array into the specified shape
    reshaped_array = values_array.reshape((rows, cols))

    # Save the reshaped array to a new file with _format extension
    np.savetxt(output_file, reshaped_array, fmt='%s')
    
    # return reshaped array
    return reshaped_array

def calculate_vertical_conductivity(vcont, thickness, nlay, nrow, ncol):
    nlay, nrow, ncol = vcont.shape
    k33 = np.zeros((nlay, nrow, ncol))

    for i in range(nlay - 1):
        for j in range(nrow):
            for k in range(ncol):
                k33[i, j, k] = vcont[i, j, k] * ((thickness[i, j, k] / 2) + (thickness[i + 1, j, k] / 2))
    # Set the vertical hydraulic conductivity of layer 12 equal to layer 11
    k33[-1] = k33[-2]
    
    return k33

# create flow model
model_name = 'dizon-mf6rtm'

# Define model parameters for the dummy model
nlay = 12  # Number of layers
nrow = 10  # Number of rows
ncol = 51  # Number of columns
nper = 39  # Number of stress periods
delc = [4, 40, 24, 16, 12,
        10, 6, 4, 4, 4]  # Column spacing (meters)
delr = [4, 16, 8, 4, 4, 4, 4, 3, 2, 2, 
        2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
        3, 4, 4, 4, 4, 4, 4, 12, 24, 40, 
        4]  # Column Spacing (meters)

perioddata= [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
             (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1), 
             (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1), 
             (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
             (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
             (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
             (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
             (28, 28, 1), (35, 35, 1), (35, 35, 1),(28, 28, 1)]

top = -273  # Top elevation (constant, meters above mean sea level)
#botm = np.linspace(40, -10, nlay)
botm = np.zeros((nlay, nrow, ncol))  # Bottom elevations of each layer (meters)
steady = [False] * nper  # All stress periods are transient

# specify the mf6 gw object & add relevant components
flow_model = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', sim_ws='.')
flow_model.set_sim_path(model_ws)

# specify tdis
tdis = flopy.mf6.ModflowTdis(flow_model, pname="tdis", time_units="DAYS", nper=nper, perioddata=perioddata)

# specify ims pkg 
ims = flopy.mf6.ModflowIms(flow_model, pname="ims", complexity="SIMPLE")

# start model build to refine 
model_nam_file = "{}.nam".format(model_name)
gwf = flopy.mf6.ModflowGwf(flow_model, modelname=model_name, model_nam_file=model_nam_file, exe_name=mf6_exe_ws_model)

# specify dis pkg
dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=delr,
    delc=delc,
    top=top,
    botm=botm,
)

# specify the initial conditions
ihead = 0 #(meters)
start = ihead * np.ones((nlay, nrow, ncol))
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

# now let's start filling data from the mf96 arrays to populate the mf6 model
layer_txt_files = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
                   'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
                   'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

# first load in mf96 top elev array reformt and assign to mf6 dis
top_elev = reformat_arrays(os.path.join(dis_ws, 'tops', 'top_elev.txt'), 
                           os.path.join(dis_ws, 'tops', 'format_top_elev.txt'), 
                           nrow, 
                           ncol)
dis.top = top_elev # meters above mean sea level

# now let's load in the mf96 bottom elevations

botm96 = np.zeros((nlay, nrow, ncol))
ibnd96 = np.zeros((nlay, nrow, ncol))
thick96 = np.zeros((nlay, nrow, ncol))
scoeff96 = np.zeros((nlay, nrow, ncol))
trans96 = np.zeros((nlay, nrow, ncol))
k = np.zeros((nlay, nrow, ncol))
k33 = np.zeros((nlay, nrow, ncol))
vcont96 = np.zeros((nlay, nrow, ncol))

for i, layer_txt_files in enumerate(layer_txt_files):
    botm96[i] = reformat_arrays(os.path.join(dis_ws, 'bottoms', layer_txt_files), 
                                os.path.join(dis_ws, 'bottoms', 'format_'+str(layer_txt_files)), 
                                nrow, 
                                ncol)
    ibnd96[i] = reformat_arrays(os.path.join(dis_ws, 'ibound', layer_txt_files), 
                                os.path.join(dis_ws, 'ibound', 'format_'+str(layer_txt_files)), 
                                nrow, 
                                ncol)
    thick96[i] = reformat_arrays(os.path.join(dis_ws, 'thicknesses', layer_txt_files), 
                                os.path.join(dis_ws, 'thicknesses', 'format_'+str(layer_txt_files)), 
                                nrow, 
                                ncol)
    scoeff96[i] = reformat_arrays(os.path.join(props_ws, 'S', layer_txt_files), 
                                os.path.join(props_ws, 'S', 'format_'+str(layer_txt_files)), 
                                nrow, 
                                ncol)
    trans96[i] = reformat_arrays(os.path.join(props_ws, 'T', layer_txt_files), 
                                os.path.join(props_ws, 'T', 'format_'+str(layer_txt_files)), 
                                nrow, 
                                ncol)
    vcont96[i] = reformat_arrays(os.path.join(props_ws, 'vcont', layer_txt_files), 
                                os.path.join(props_ws, 'vcont', 'format_'+str(layer_txt_files)), 
                                nrow, 
                                ncol)
    
dis.botm = botm96 # meters above mean sea level
dis.idomain = ibnd96
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=0,
    k=k,
    k33=k33,
)
npf.k = trans96/thick96
  
# Calculate vertical hydraulic conductivity from vcont
k33_array = calculate_vertical_conductivity(vcont96, thick96, nlay, nrow, ncol)
k33 = flopy.mf6.ModflowGwfnpf(gwf, k33=k33_array)

# Create the sto package with specific storage and specific yield
sto = flopy.mf6.ModflowGwfsto(gwf, ss=scoeff96/thick96, iconvert=0)

# Define wel package injection & extraction wells & schedules

flopy_offset = 1
pumping_data_init = [[2-flopy_offset, 10-flopy_offset, 10-flopy_offset, -300],
                     [2-flopy_offset, 10-flopy_offset, 39-flopy_offset, 117.5],
                     [3-flopy_offset, 10-flopy_offset, 39-flopy_offset, 40],
                     [4-flopy_offset, 10-flopy_offset, 10-flopy_offset, -30],
                     [4-flopy_offset, 10-flopy_offset, 39-flopy_offset, 70],
                     [6-flopy_offset, 10-flopy_offset, 10-flopy_offset, -30],
                     [6-flopy_offset, 10-flopy_offset, 39-flopy_offset, 95],
                     [8-flopy_offset, 10-flopy_offset, 39-flopy_offset, 37.5]]

pumping_data_fini = [[2-flopy_offset, 10-flopy_offset, 10-flopy_offset, -400],
                     [2-flopy_offset, 10-flopy_offset, 39-flopy_offset, 156.7],
                     [3-flopy_offset, 10-flopy_offset, 39-flopy_offset, 53.3],
                     [4-flopy_offset, 10-flopy_offset, 10-flopy_offset, -40],
                     [4-flopy_offset, 10-flopy_offset, 39-flopy_offset, 93.3],
                     [6-flopy_offset, 10-flopy_offset, 10-flopy_offset, -40],
                     [6-flopy_offset, 10-flopy_offset, 39-flopy_offset, 126.6],
                     [8-flopy_offset, 10-flopy_offset, 39-flopy_offset, 50]]

stress_period_data = {0: pumping_data_init, 1: pumping_data_init, 2: pumping_data_init,
                      3: pumping_data_init, 4: pumping_data_init, 5: pumping_data_init,
                      6: pumping_data_init, 7: pumping_data_init, 8: pumping_data_init,
                      9: pumping_data_init, 10: pumping_data_init, 11: pumping_data_init,
                      12: pumping_data_init, 13: pumping_data_init, 14: pumping_data_init,
                      15: pumping_data_init, 16: pumping_data_init, 17: pumping_data_init,
                      18: pumping_data_init, 19: pumping_data_init, 20: pumping_data_init,
                      21: pumping_data_init, 22: pumping_data_init, 23: pumping_data_init,
                      24: pumping_data_init, 25: pumping_data_init, 26: pumping_data_init,
                      27: pumping_data_init, 28: pumping_data_init, 29: pumping_data_init,
                      30: pumping_data_init, 31: pumping_data_init, 32: pumping_data_init,
                      33: pumping_data_init,  34: pumping_data_init, 35: pumping_data_init,
                      36: pumping_data_fini, 37: pumping_data_fini, 38: pumping_data_fini}

wel = flopy.mf6.ModflowGwfwel(gwf,stress_period_data=stress_period_data, filename='dizon-mf6rtm.wel')

# create the output control
headfile = f"{model_name}.hds"
head_filerecord = [headfile]
budgetfile = f"{model_name}.cbb"
budget_filerecord = [budgetfile]
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
printrecord = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    saverecord=saverecord,
    head_filerecord=head_filerecord,
    budget_filerecord=budget_filerecord,
    printrecord=printrecord,)

# write the mf6 flow model input files
flow_model.write_simulation()

# Run the mf6 flow simulation
os.chdir(model_ws)
# Run the executable from the destination folder
result = subprocess.run(['mf6.exe'], check=True, capture_output=True, text=True)

# Capture standard output and error messages from the executable
print("Simulation Output:")
print(result.stdout)

# Check if the simulation ran successfully
if result.returncode == 0:
    print("Simulation ran successfully!")
else:
    print(f"Simulation failed with error code {result.returncode}.")
    print(result.stderr)
    
os.chdir(cwd)
    
# Get the time discretization information
tdis = gwf.simulation.tdis
perioddata = tdis.perioddata.array

# Calculate cumulative end times from each stress period
cumulative_time = 0
end_of_stress_period_times = []
for perlen in perioddata["perlen"]:
    cumulative_time += perlen
    end_of_stress_period_times.append(cumulative_time)

# plot heads in mapview for each layer
h = gwf.output.head().get_alldata()

# plot the head results mapview
mapview_output_dir = os.path.join(model_ws, 'output', 'mapview')
if not os.path.exists(mapview_output_dir):
    os.makedirs(mapview_output_dir)

mapview_counter = 0
for t in range(nper-1, nper):
    h_t = h[t, :, :, :]
    masked_h = np.ma.masked_where((h_t < -500) | (h_t > 500), h_t)
    vmin = masked_h.min()
    vmax = masked_h.max()
    vmin, vmax = np.nanmin(masked_h), np.nanmax(masked_h)  # Set colorbar limits

    contour_intervals = np.linspace(vmin, vmax, 10)  # Contour levels

    #contour_intervals = np.arange(vmin, vmax, (vmax-vmin)/10)
    
    # Create a figure with 4 rows and 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(16, 16), constrained_layout=True)
    fig.subplots_adjust(right=0.85)  # Leave space for the colorbar
    
    # Loop over all layers
    for i, ax in enumerate(axes.flat):
        if i < nlay:
            # Plot for layer `i`
            ax.set_title(f"Model Layer {i + 1}")
            modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=i)
    
            # Plot array, boundary conditions, and contours
            pa = modelmap.plot_array(masked_h[i, :, :], vmin=vmin, vmax=vmax)
            modelmap.plot_bc("WEL", color='black')
            modelmap.plot_grid(lw=0.5, color="0.5")
            contours = modelmap.contour_array(
                masked_h[i, :, :],
                levels=contour_intervals,
                colors="black",
            )
            ax.clabel(contours, fmt="%2.1f")
        else:
            # Hide axes for extra subplots
            ax.axis("off")
    
    # Add a single colorbar for the entire figure
    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(pa, cax=cbar_ax, label="Head (mamsl)")
    plt.suptitle('Simulated Head at Stress Period: ' + str(t+1) + ' time in days: ' + str(end_of_stress_period_times[t]))
    # Use tight_layout for clean subplot spacing
    #fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to leave space for the colorbar
    fig.savefig(os.path.join(os.getcwd(), mapview_output_dir, '000' + str(mapview_counter) + '_SimulatedHeads_mamsl_nper_' + str(t+1) + '_time_days_' + str(end_of_stress_period_times[t]) + '.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)
    mapview_counter+=1
    #plt.show()

# plot the head results xsect
xsect_output_dir = os.path.join(model_ws, 'output', 'xsect')
if not os.path.exists(xsect_output_dir):
    os.makedirs(xsect_output_dir)
    
# plot heads in each row as a cross-section through all 12 layers
for t in range(nper):
    h_t = h[t, :, :, :]
    masked_h = np.ma.masked_where((h_t < -500) | (h_t > 500), h_t)
    vmin = masked_h.min()
    vmax = masked_h.max()
    vmin, vmax = np.nanmin(masked_h), np.nanmax(masked_h)  # Set colorbar limits
    
    for r in range(nrow-1, nrow):
        fig, ax = plt.subplots(1, 1, figsize=(9, 3), constrained_layout=True)
        # first subplot
        ax.set_title("Row: " + f"{r+1}" + " Stress Period - nper: " + f"{t}" + ' time in days: ' + str(end_of_stress_period_times[t]))
        modelmap = flopy.plot.PlotCrossSection(
            model=gwf,
            ax=ax,
            line={"row": f"{r}"},
        )
        pa = modelmap.plot_array(masked_h, vmin=vmin, vmax=vmax)
        quadmesh = modelmap.plot_bc("WEL", color='red')
        linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
        contours = modelmap.contour_array(
            masked_h,
            levels=contour_intervals,
            colors="black",
        )
        ax.clabel(contours, fmt="%2.1f")
        cb = plt.colorbar(pa, shrink=0.5, ax=ax)
        cb.set_label('head (mamsl)')
        
        #plt.tight_layout()
        fig.savefig(os.path.join(xsect_output_dir, '00' + str(r+1) + '_' + str(t+1) + '_SimulatedHeads_mamsl_nrow_' + str(r+1) + '_nper_' + str(t+1) + '_time_days_' + str(end_of_stress_period_times[t]) + '.png'), dpi=400, bbox_inches='tight')
        #plt.show()
    
# transport model parameters
ne = 0.35 # effective porosity (-) constant across all layers
long_disp = 0.01 # Longitudinal dispersivity (m)constant across all layers
disp_tr_vert = long_disp*0.01 # Transverse vertical dispersivity (m) constant across all layers
disp_tr_hor = long_disp*0.1 # Transverse horizontal dispersivity (m) constant across all layers
diffc = 0 # diffusion coefficient constant across all layers
pbulk = 1850 # bulk density in constant across all layers (m/L^3)

transport_parameters = {

    'ne':           [ne, ne, ne, ne, ne, ne,
                     ne, ne, ne, ne, ne, ne],
    'long_disp':    [long_disp, long_disp, long_disp, long_disp, long_disp, long_disp,
                     long_disp, long_disp, long_disp, long_disp, long_disp, long_disp],
    'disp_tr_vert': [disp_tr_vert, disp_tr_vert, disp_tr_vert, disp_tr_vert, disp_tr_vert, disp_tr_vert, 
                     disp_tr_vert, disp_tr_vert, disp_tr_vert, disp_tr_vert, disp_tr_vert, disp_tr_vert],
    'disp_tr_hor':  [disp_tr_hor, disp_tr_hor, disp_tr_hor, disp_tr_hor, disp_tr_hor, disp_tr_hor, 
                     disp_tr_hor, disp_tr_hor, disp_tr_hor, disp_tr_hor, disp_tr_hor, disp_tr_hor],
    'diffc':        [diffc, diffc, diffc, diffc, diffc, diffc, 
                     diffc, diffc, diffc, diffc, diffc, diffc],
    'pbulk':        [pbulk, pbulk, pbulk, pbulk, pbulk, pbulk, 
                     pbulk, pbulk, pbulk, pbulk, pbulk, pbulk]
}

lin_sorp_distr_coeff = {

    'Orgc':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'O(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'C(4)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'C(-4)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Ca':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Cl':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    'Fe(2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Fe(3)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'K':            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Mg':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'N(3)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'N(5)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Na':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'S(-2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'S(6)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Si':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Amm':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'N(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Tmp':          [2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04,
                     2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04],
    'pH':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'pe':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Ferrihydrite': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Orgmatter':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Ca_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Fe_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'K_ex':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Mg_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Na_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Pyrite':       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

}

# define initial concentration conditions in all 12 layers
initial_concentrations = {

    'Orgc':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'O(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'C(4)':         [0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 
                     0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 0.008446],
    'C(-4)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Ca':           [0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 
                     0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 0.002057],
    'Cl':           [0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 
                     0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 0.000254], 
    'Fe(2)':        [0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 
                     0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041],
    'Fe(3)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'K':            [0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336,
                     0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336],
    'Mg':           [0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 
                     0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875],
    'N(3)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'N(5)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Na':           [0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 
                     0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748],
    'S(-2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'S(6)':         [5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05,
                     5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05],
    'Si':           [0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 
                     0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497],
    'Amm':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'N(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Tmp':          [0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 
                     0.017, 0.017, 0.017, 0.017, 0.017, 0.017],
    'pH':           [6.601, 6.601, 6.601, 6.601, 6.601, 6.601, 
                     6.601, 6.601, 6.601, 6.601, 6.601, 6.681],
    'pe':           [-2.449, -2.449, -2.449, -2.449, -2.449, -2.449, 
                     -2.449, -2.449, -2.449, -2.449, -2.449, -2.449],
    'Ferrihydrite': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Orgmatter':    [2.410781, 0.3873438, 1.445313, 0.1907813, 0.1907813, 0.1907813,
                     1.925156, 1.925156, 1.925156, 1.925156, 1.925156, 1.925156],
    'Ca_ex':        [0.1188, 0.02138, 0.07358, 0.01891, 0.01891, 0.01891,
                     0.05919, 0.05919, 0.05919, 0.05919, 0.05919, 0.05919],
    'Fe_ex':        [0.002451, 0.0004411, 0.001518, 0.0003902, 0.0003902, 0.0003902,
                     0.001221, 0.001221, 0.001221, 0.001221, 0.001221, 0.001221],
    'K_ex':         [0.00157, 0.0002825, 0.0009723, 0.0002499, 0.0002499, 0.0002499, 
                     0.0007822, 0.0007822, 0.0007822, 0.0007822, 0.0007822, 0.0007822],
    'Mg_ex':        [0.02131, 0.003835, 0.0132, 0.003393, 0.003393, 0.003393, 
                     0.01062, 0.01062, 0.01062, 0.01062, 0.01062, 0.01062],
    'Na_ex':        [0.001596, 0.0002872, 0.0009888, 0.0002541, 0.0002541, 0.0002541,
                     0.0007954, 0.0007954, 0.0007954, 0.0007954, 0.0007954, 0.0007954],
    'Pyrite':       [0.0629, 0.0222, 0.0999, 0.01295, 0.01295, 0.01295, 
                     0.13135, 0.13135, 0.13135, 0.13135, 0.13135, 0.13135]
    
}

# define icbund arrays for each layer for the transport model from the previous pht3d model
layer_txt_files = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
                   'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
                   'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

transport_ws = os.path.join(os.getcwd(), 'transport')

pht3d_icbund = np.zeros((nlay, nrow, ncol))
for i, layer_txt_files in enumerate(layer_txt_files):
    pht3d_icbund[i] = reformat_arrays(os.path.join(transport_ws, 'props', 'icbund', layer_txt_files), 
                                os.path.join(transport_ws, 'props', 'icbund', 'format_'+str(layer_txt_files)), 
                                nrow, 
                                ncol)

# import phinp.dat from pht3d model using phreeqc rm
nxyz = nlay * ncol * nrow
nthreads = 3

phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz, nthreads)

database_fpth = os.path.join(os.getcwd(), 'transport', 'database', 'dizon.pht3d_database')

# Load the database
phreeqc_rm.LoadDatabase(database_fpth)

pqi_fpth = os.path.join(os.getcwd(), 'transport', 'pqi', 'phinp.dat')

# Run the input file
phreeqc_rm.RunFile(True, True, True, pqi_fpth)