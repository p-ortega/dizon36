# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:48:41 2025

@author: afoster
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy

cwd = os.getcwd()
dis_ws = os.path.join(cwd, 'dis')
props_ws = os.path.join(cwd, 'props')
welpkg_ws = os.path.join(cwd, 'wel pkg')

model_ws = os.path.join(os.getcwd(), 'model')
if not os.path.exists(model_ws):
    os.makedirs(model_ws)

# # Read the file and parse the numbers into a list
# with open(os.path.join(dis_ws, 'top_elev_mamsl.txt'), "r") as file:
#     numbers = [float(num) for line in file for num in line.split()]

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

def read_wel_package(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip the binary header line
    lines = lines[1:]

    # Read and skip the max number of wells from the second line
    max_wells = int(lines[0].strip().split()[0])
    lines = lines[1:]

    stress_periods = []
    current_stress_period = []

    for line in lines:
        line = line.strip()
        if line == '-1':
            stress_periods.append(current_stress_period)
            current_stress_period = []
        else:
            current_stress_period.append(list(map(float, line.split())))

    return stress_periods

def convert_wel_to_mf6(stress_periods, nper):
    wel_data = {}

    for per in range(nper):
        if per < len(stress_periods):
            period_data = stress_periods[per]
            wel_data[per] = [[int(rec[0]) - 1, int(rec[1]) - 1, int(rec[2]) - 1, rec[3]] for rec in period_data]
        else:
            wel_data[per] = []

    return wel_data

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

# specify minimum packages to get a placeholder model set-up to fill in relevant model data

top = -273  # Top elevation (constant, meters above mean sea level)
#botm = np.linspace(40, -10, nlay)
botm = np.zeros((nlay, nrow, ncol))  # Bottom elevations of each layer (meters)
steady = [False] * nper  # All stress periods are transient

# specify the mf6 gw object & add relevant components
flow_model = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', sim_ws='.')

# specify tdis
tdis = flopy.mf6.ModflowTdis(flow_model, pname="tdis", time_units="DAYS", nper=nper, perioddata=perioddata)

# specify ims pkg 
ims = flopy.mf6.ModflowIms(flow_model, pname="ims", complexity="SIMPLE")

# start model build to refine 
model_nam_file = "{}.nam".format(model_name)
gwf = flopy.mf6.ModflowGwf(flow_model, modelname=model_name, model_nam_file=model_nam_file)

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
for i, layer_txt_files in enumerate(layer_txt_files):
    botm96[i] = reformat_arrays(os.path.join(dis_ws, 'bottoms', layer_txt_files), 
                                os.path.join(dis_ws, 'bottoms', 'format_'+str(layer_txt_files)), 
                                nrow, 
                                ncol)
dis.botm = botm96 # meters above mean sea level

# now let's start filling data from the mf96 arrays to populate the mf6 model
layer_txt_files_2 = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
                   'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
                   'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

ibnd96 = np.zeros((nlay, nrow, ncol))
for j, layer_txt_files_2 in enumerate(layer_txt_files_2):
    ibnd96[j] = reformat_arrays(os.path.join(dis_ws, 'ibound', layer_txt_files_2), 
                                os.path.join(dis_ws, 'ibound', 'format_'+str(layer_txt_files_2)), 
                                nrow, 
                                ncol)
dis.idomain = ibnd96

# now let's start filling data from the mf96 arrays to populate the mf6 model
layer_txt_files_3 = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
                   'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
                   'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

thick96 = np.zeros((nlay, nrow, ncol))
for j, layer_txt_files_3 in enumerate(layer_txt_files_3):
    thick96[j] = reformat_arrays(os.path.join(dis_ws, 'thicknesses', layer_txt_files_3), 
                                os.path.join(dis_ws, 'thicknesses', 'format_'+str(layer_txt_files_3)), 
                                nrow, 
                                ncol)
    
# now let's start filling data from the mf96 arrays to populate the mf6 model
layer_txt_files_4 = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
                   'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
                   'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

scoeff96 = np.zeros((nlay, nrow, ncol))
for j, layer_txt_files_4 in enumerate(layer_txt_files_4):
    scoeff96[j] = reformat_arrays(os.path.join(props_ws, 'S', layer_txt_files_4), 
                                os.path.join(props_ws, 'S', 'format_'+str(layer_txt_files_4)), 
                                nrow, 
                                ncol)
    
# now let's start filling data from the mf96 arrays to populate the mf6 model
layer_txt_files_5 = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
                   'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
                   'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

trans96 = np.zeros((nlay, nrow, ncol))
for j, layer_txt_files_5 in enumerate(layer_txt_files_5):
    trans96[j] = reformat_arrays(os.path.join(props_ws, 'T', layer_txt_files_5), 
                                os.path.join(props_ws, 'T', 'format_'+str(layer_txt_files_5)), 
                                nrow, 
                                ncol)

k = np.zeros((nlay, nrow, ncol))
k33 = np.zeros((nlay, nrow, ncol))
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=0,
    k=k,
    k33=k33,
)
npf.k = trans96/thick96

# now let's start filling data from the mf96 arrays to populate the mf6 model
layer_txt_files_6 = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
                   'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
                   'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

vcont96 = np.zeros((nlay, nrow, ncol))
for j, layer_txt_files_6 in enumerate(layer_txt_files_6):
    vcont96[j] = reformat_arrays(os.path.join(props_ws, 'vcont', layer_txt_files_6), 
                                os.path.join(props_ws, 'vcont', 'format_'+str(layer_txt_files_6)), 
                                nrow, 
                                ncol)
    
# Calculate vertical hydraulic conductivity from vcont
k33 = calculate_vertical_conductivity(vcont96, thick96, nlay, nrow, ncol)

# Create the sto package with specific storage and specific yield
sto = flopy.mf6.ModflowGwfsto(gwf, ss=scoeff96/thick96, iconvert=0)

# let's read in the old wel package and reformat to modflow 6
# Example usage
#file_path = 'wel_package.txt'
#nper = 39

# Read the MODFLOW 96 wel package input file
#stress_periods = read_wel_package(os.path.join(welpkg_ws, 'wel-test.dat'))

# Convert the wel package data to MODFLOW 6 format
#wel_data_mf6 = convert_wel_to_mf6(stress_periods, nper)

# MF96 model defined all units as meters and days
flow_model.write_simulation(model_ws=model_ws)

# Run the simulation
success, buff = flow_model.run_simulation()

if success:
    print("Simulation ran successfully!")
else:
    print("Simulation failed. Check the output for errors.")

# transport model parameters
ne = 0.35 # effective porosity (-)


