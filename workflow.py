import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import itertools
from mf6rtm import utils
from collections import defaultdict

datadir = os.path.join("data")
dis_ws = os.path.join(datadir, 'dis')
props_ws = os.path.join(datadir, 'props')

def run_model(sim):
    import pyemu
    pyemu.os_utils.run('mf6', cwd=sim.sim_path)

def reformat_arrays(input_file, rows, cols, save_file=False):
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
    if save_file:
        root, ext = os.path.splitext(input_file)
        output_file = f"{root}_formatted{ext}"
        print(f"Saving reformated file to {output_file}")
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

def make_dis(gwf):
    # Define model parameters for the dummy model
    nlay = 12  # Number of layers
    nrow = 10  # Number of rows
    ncol = 51  # Number of columns
    ib = np.ones((nlay, nrow, ncol))
    delc = [4, 40, 24, 16, 12,
            10, 6, 4, 4, 4]  # Column spacing (meters)
    delr = [4, 16, 8, 4, 4, 4, 4, 3, 2, 2, 
            2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
            3, 4, 4, 4, 4, 4, 4, 12, 24, 40, 
            4]  # Column Spacing (meters)
    top = -273  # Top elevation (constant, meters above mean sea level)
    botm = np.zeros((nlay, nrow, ncol))  # Bottom elevations of each layer (meters)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        idomain=ib,
        botm=botm,
    )
    top_elev = reformat_arrays(os.path.join(dis_ws, 'tops', 'top_elev.txt'), 
                        nrow, 
                        ncol)
    
    layer_txt_files = [f"layer_{i}.txt" for i in range(1, 13)]

    botm96 = [
        reformat_arrays(os.path.join(dis_ws, 'bottoms', filename), nrow, ncol)
        for filename in layer_txt_files
    ]

    dis.top = top_elev # meters above mean sea level
    dis.botm = botm96 # meters above mean sea level
    dis.set_all_data_external()
    return dis

def make_gwf(model_name = "gwf"):
    model_ws = os.path.join("model")
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)

    # Re‑create an empty folder
    os.makedirs(model_ws, exist_ok=True)
    utils.prep_bins(model_ws)

    nper = 39  # Number of stress periods

    perioddata= [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
                (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1), 
                (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1), 
                (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (35, 35, 1), (35, 35, 1),(28, 28, 1)]

    steady = [False] * nper  # All stress periods are transient

    # specify the mf6 gw object & add relevant components
    sim = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', sim_ws='.')
    sim.set_sim_path(model_ws)

    # specify tdis
    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS", 
                                 nper=nper, perioddata=perioddata)

    ims = flopy.mf6.ModflowIms(sim, 
                            #    pname="ims", 
                            complexity="COMPLEX",
                            filename=f"{model_name}.ims")
    sim.register_ims_package(ims, 
                             [model_name])
    
    # start model build to refine 
    model_nam_file = "{}.nam".format(model_name)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, 
                               model_nam_file=model_nam_file, exe_name='mf6')

    # make dis
    dis = make_dis(gwf)

    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()

    # get properties
    botm96 = np.zeros((nlay, nrow, ncol))
    
    thick96 = np.zeros((nlay, nrow, ncol))
    scoeff96 = np.zeros((nlay, nrow, ncol))
    trans96 = np.zeros((nlay, nrow, ncol))
    k = np.zeros((nlay, nrow, ncol))
    k33 = np.zeros((nlay, nrow, ncol))
    vcont96 = np.zeros((nlay, nrow, ncol))

    # now let's start filling data from the mf96 arrays to populate the mf6 model
    layer_txt_files = [f"layer_{i}.txt" for i in range(1, 13)]

    
    for i, layer_txt_files in enumerate(layer_txt_files):

        thick96[i] = reformat_arrays(os.path.join(dis_ws, 'thicknesses', layer_txt_files), 
                                    nrow, 
                                    ncol)
        scoeff96[i] = reformat_arrays(os.path.join(props_ws, 's', layer_txt_files), 
                                    nrow, 
                                    ncol)
        trans96[i] = reformat_arrays(os.path.join(props_ws, 'T', layer_txt_files), 
                                    nrow, 
                                    ncol)
        vcont96[i] = reformat_arrays(os.path.join(props_ws, 'vcont', layer_txt_files), 
                                    nrow, 
                                    ncol)

    # first load in mf96 top elev array reformt and assign to mf6 dis

    ihead = 0 #(meters)
    start = ihead * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)
    ic.set_all_data_external()

    karr=trans96/thick96
    k33_array = calculate_vertical_conductivity(vcont96, thick96, nlay, nrow, ncol)

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=karr,
        k33=k33_array,
    )
    # npf.k = trans96/thick96
    npf.set_all_data_external()

    sto = flopy.mf6.ModflowGwfsto(gwf, ss=scoeff96/thick96, iconvert=0)
    sto.set_all_data_external()

    # CHD boundaries
    l_hd= 0
    chdspd = []

    for i in range(nlay):          # layers
        for j in range(nrow):      # rows
            chdspd.append([(i, j, 0), l_hd])           # left boundary
            chdspd.append([(i, j, ncol - 1), l_hd])    # right boundary

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=True,
        pname="CHD",
        filename=f"{model_name}.chd",
    )
    chd.set_all_data_external()

    coords_out = [ # (row, col, layer) 
        (1, 9,  9),
        (3, 9,  9),
        (5, 9,  9),
    ]

    coords_in = [
        (1, 9, 38),
        (2, 9, 38),
        (3, 9, 38),
        (5, 9, 38),
        (7, 9, 38),
    ]

    init_rates_out  = [-300,  -30,  -30]                # 3 negatives
    init_rates_in   = [117.5,  40,  70,  95,  37.5]     # 5 positives

    fini_rates_out  = [-400,  -40,  -40]
    fini_rates_in   = [156.7,  53.3,  93.3, 126.6,  50]

    init_sp = range(0, 36)   # stress periods 0 – 35
    fini_sp = range(36, 39)  # stress periods 36 – 38
    all_sp  = (*init_sp, *fini_sp)

    df_inj = pd.read_csv(os.path.join(datadir, "wellin.csv"))
    
    wellin_sp_data = defaultdict(list)

    for _, r in df_inj.iterrows():
        cell = (int(r["layer"]), int(r["row"]), int(r["column"]))  # zero‑indexed
        wellin_sp_data[int(r["kper"])].append([cell, r["rate"], r["Cl"]])

    def make_rows(coords, rates, add_conc=True):
        if len(coords) != len(rates):
            raise ValueError("Coordinate and rate lists must be the same length")
        return [
            ([cell, q, 0.0] if add_conc else [cell, q])
            for cell, q in zip(coords, rates)
        ]

    # Time‑invariant blocks for each phase
    # wellin_init   = make_rows(coords_in,  init_rates_in)
    wellout_init  = make_rows(coords_out, init_rates_out)
    # wellin_fini   = make_rows(coords_in,  fini_rates_in,  add_conc=True)
    wellout_fini  = make_rows(coords_out, fini_rates_out, add_conc=True)

    # 3) Assemble stress‑period dictionaries
    # wellin_sp_data  = {sp: (wellin_init  if sp in init_sp else wellin_fini)
    #                 for sp in all_sp}

    wellout_sp_data = {sp: (wellout_init if sp in init_sp else wellout_fini)
                    for sp in all_sp}
    
    wel_in  = flopy.mf6.ModflowGwfwel(gwf, 
                                       stress_period_data=wellin_sp_data,
                                       auxiliary='Cl',
                                       pname = 'welin',
                                       filename=f'{model_name}.welin')
    wel_in.set_all_data_external()
    wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                       stress_period_data=wellout_sp_data, 
                                       auxiliary='Cl',
                                       pname = 'welout' ,
                                       filename=f'{model_name}.welout')
    wel_out.set_all_data_external()
    # Define wel package injection & extraction wells & schedules
    # flopy_offset = 1
    # pumping_data_init = [[(2-flopy_offset, 10-flopy_offset, 10-flopy_offset), -300 , 0],
    #                     [(2-flopy_offset, 10-flopy_offset, 39-flopy_offset), 117.5, 0],
    #                     [3-flopy_offset, 10-flopy_offset, 39-flopy_offset, 40, 0],
    #                     [4-flopy_offset, 10-flopy_offset, 10-flopy_offset, -30, 0],
    #                     [4-flopy_offset, 10-flopy_offset, 39-flopy_offset, 70],
    #                     [6-flopy_offset, 10-flopy_offset, 10-flopy_offset, -30],
    #                     [6-flopy_offset, 10-flopy_offset, 39-flopy_offset, 95],
    #                     [8-flopy_offset, 10-flopy_offset, 39-flopy_offset, 37.5]]

    # pumping_data_fini = [[2-flopy_offset, 10-flopy_offset, 10-flopy_offset, -400],
    #                     [2-flopy_offset, 10-flopy_offset, 39-flopy_offset, 156.7],
    #                     [3-flopy_offset, 10-flopy_offset, 39-flopy_offset, 53.3],
    #                     [4-flopy_offset, 10-flopy_offset, 10-flopy_offset, -40],
    #                     [4-flopy_offset, 10-flopy_offset, 39-flopy_offset, 93.3],
    #                     [6-flopy_offset, 10-flopy_offset, 10-flopy_offset, -40],
    #                     [6-flopy_offset, 10-flopy_offset, 39-flopy_offset, 126.6],
    #                     [8-flopy_offset, 10-flopy_offset, 39-flopy_offset, 50]]

    # stress_period_data = {0: pumping_data_init, 1: pumping_data_init, 2: pumping_data_init,
    #                     3: pumping_data_init, 4: pumping_data_init, 5: pumping_data_init,
    #                     6: pumping_data_init, 7: pumping_data_init, 8: pumping_data_init,
    #                     9: pumping_data_init, 10: pumping_data_init, 11: pumping_data_init,
    #                     12: pumping_data_init, 13: pumping_data_init, 14: pumping_data_init,
    #                     15: pumping_data_init, 16: pumping_data_init, 17: pumping_data_init,
    #                     18: pumping_data_init, 19: pumping_data_init, 20: pumping_data_init,
    #                     21: pumping_data_init, 22: pumping_data_init, 23: pumping_data_init,
    #                     24: pumping_data_init, 25: pumping_data_init, 26: pumping_data_init,
    #                     27: pumping_data_init, 28: pumping_data_init, 29: pumping_data_init,
    #                     30: pumping_data_init, 31: pumping_data_init, 32: pumping_data_init,
    #                     33: pumping_data_init,  34: pumping_data_init, 35: pumping_data_init,
    #                     36: pumping_data_fini, 37: pumping_data_fini, 38: pumping_data_fini}

    # wel = flopy.mf6.ModflowGwfwel(gwf,
    #                               stress_period_data=stress_period_data, 
    #                               auxiliary='gwt',
    #                               filename=f'{model_name}.wel')
    # wel.set_all_data_external()

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
    # sim.write_simulation()
    return sim


######## transport ###############

def make_gwt(sim, model_name = 'gwt'):

    gwf = sim.get_model("gwf")
    nlay = gwf.dis.nlay.get_data()

    ne = 0.35 # effective porosity (-) constant across all layers
    long_disp = 0.01 # Longitudinal dispersivity (m)constant across all layers
    disp_tr_vert = long_disp*0.01 # Transverse vertical dispersivity (m) constant across all layers
    disp_tr_hor = long_disp*0.1 # Transverse horizontal dispersivity (m) constant across all layers
    diffc = 0 # diffusion coefficient constant across all layers
    pbulk = 1850 # bulk density in constant across all layers (m/L^3) 


    ne = 0.35              # effective porosity (-)
    long_disp = 0.01       # longitudinal dispersivity (m)
    disp_tr_vert = long_disp * 0.01  # transverse vertical dispersivity (m)
    disp_tr_hor = long_disp * 0.1    # transverse horizontal dispersivity (m)
    diffc = 0              # diffusion coefficient
    pbulk = 1850           # bulk density (m/L^3)

    # transport_parameters = {
    #     'ne':           [ne] * nlay,
    #     'long_disp':    [long_disp] * nlay,
    #     'disp_tr_vert': [disp_tr_vert] * nlay,
    #     'disp_tr_hor':  [disp_tr_hor] * nlay,
    #     'diffc':        [diffc] * nlay,
    #     'pbulk':        [pbulk] * nlay,
    # }

    # print(transport_parameters)

    gwt = flopy.mf6.MFModel(
        sim,
        model_type="gwt6",
        modelname=model_name,
        model_nam_file=f"{model_name}.nam"
    )

    imsgwt = flopy.mf6.ModflowIms(sim, 
                                #   pname="ims", 
                            complexity="COMPLEX",
                            filename=f"{model_name}.ims")
    sim.register_ims_package(imsgwt, 
                             [model_name])
    
    # nper = sim.tdis.nper.get_data()
    # perioddata = sim.tdis.perioddata.get_data()
    # start_date_time = sim.tdis.start_date_time.get_data()

    # tdis = flopy.mf6.ModflowTdis(sim, pname="tdis",
    #                                 nper=nper, 
    #                                 perioddata=perioddata, #gwt_perioddata,
    #                                 time_units='days', 
    #                                 start_date_time=start_date_time)

    dis = gwf.dis

    dis = flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=gwf.dis.nlay.get_data(),
            nrow=gwf.dis.nrow.get_data(),
            ncol=gwf.dis.ncol.get_data(),
            delr=gwf.dis.delr.get_data(),
            delc=gwf.dis.delc.get_data(),
            top=gwf.dis.top.get_data(),
            botm=gwf.dis.botm.get_data(),
            idomain=gwf.dis.idomain.get_data(),
            filename=f"{model_name}.dis",
        )
    dis.set_all_data_external()

    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()

    strt = np.ones(shape=(nlay,ncol, nrow))*0.000254
    ic = flopy.mf6.ModflowGwtic(gwt, strt=strt, 
                                filename=f"{model_name}.ic")
    ic.set_all_data_external()

    adv = flopy.mf6.ModflowGwtadv(
        gwt,
        scheme="tvd",
    )
    adv.set_all_data_external()

    alpha_l = np.ones(shape=(nlay,ncol, nrow))*long_disp  # Longitudinal dispersivity ($m$)
    alpha_th = np.ones(shape=(nlay,ncol, nrow))*disp_tr_hor  # Transverse horizontal dispersivity ($m$)
    alpha_tv = np.ones(shape=(nlay,ncol, nrow))*disp_tr_vert  # Transverse vertical dispersivity ($m$)

    dsp = flopy.mf6.ModflowGwtdsp(
        gwt,
        xt3d_off=True,
        alh=alpha_l,
        ath1=alpha_th,
        atv = alpha_tv,
        diffc = diffc,
        filename=f"{model_name}.dsp",
    )
    dsp.set_all_data_external()

    sourcerecarray = [["welin", "aux", f"Cl"],
                      ["welout", "aux", f"Cl"]]

    ssm = flopy.mf6.ModflowGwtssm(
            gwt,
            sources=sourcerecarray,
            save_flows=True,
            print_flows=True,
            filename=f"{model_name}.ssm",
        )
    ssm.set_all_data_external()

    mst = flopy.mf6.ModflowGwtmst(
        gwt,
        porosity=ne,
        first_order_decay=None,
        decay = None,
        decay_sorbed=None,
        sorption= None,
        bulk_density=pbulk, 
        distcoef=None, #Kd m3/mg
        sp2 = None,
        filename=f"{model_name}.mst",
    )
    mst.set_all_data_external()
    oc = flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord=f"{model_name}.cbb",
        concentration_filerecord=f"{model_name}.ucn",
        concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 10, "GENERAL")
                                    ],
        saverecord=[("CONCENTRATION", "ALL"), 
                    ],
        printrecord=[("CONCENTRATION", "LAST"), 
                        ],
    )
    flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea='gwf',
        exgmnameb=f'{model_name}',
        filename=f"{model_name}.gwfgwt",
    )
    
    sim.write_simulation() 
    return sim
# lin_sorp_distr_coeff = {

#     'Orgc':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'O(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'C(4)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'C(-4)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Ca':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Cl':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#     'Fe(2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Fe(3)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'K':            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Mg':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(3)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(5)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Na':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'S(-2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'S(6)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Si':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Amm':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Tmp':          [2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04,
#                      2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04],
#     'pH':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'pe':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Ferrihydrite': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Orgmatter':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Ca_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Fe_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'K_ex':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Mg_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Na_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Pyrite':       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

# }

# # define initial concentration conditions in all 12 layers
# initial_concentrations = {

#     'Orgc':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'O(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'C(4)':         [0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 
#                      0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 0.008446],
#     'C(-4)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Ca':           [0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 
#                      0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 0.002057],
#     'Cl':           [0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 
#                      0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 0.000254], 
#     'Fe(2)':        [0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 
#                      0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041],
#     'Fe(3)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'K':            [0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336,
#                      0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336],
#     'Mg':           [0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 
#                      0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875],
#     'N(3)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(5)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Na':           [0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 
#                      0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748],
#     'S(-2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'S(6)':         [5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05,
#                      5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05],
#     'Si':           [0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 
#                      0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497],
#     'Amm':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Tmp':          [0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 
#                      0.017, 0.017, 0.017, 0.017, 0.017, 0.017],
#     'pH':           [6.601, 6.601, 6.601, 6.601, 6.601, 6.601, 
#                      6.601, 6.601, 6.601, 6.601, 6.601, 6.681],
#     'pe':           [-2.449, -2.449, -2.449, -2.449, -2.449, -2.449, 
#                      -2.449, -2.449, -2.449, -2.449, -2.449, -2.449],
#     'Ferrihydrite': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Orgmatter':    [2.410781, 0.3873438, 1.445313, 0.1907813, 0.1907813, 0.1907813,
#                      1.925156, 1.925156, 1.925156, 1.925156, 1.925156, 1.925156],
#     'Ca_ex':        [0.1188, 0.02138, 0.07358, 0.01891, 0.01891, 0.01891,
#                      0.05919, 0.05919, 0.05919, 0.05919, 0.05919, 0.05919],
#     'Fe_ex':        [0.002451, 0.0004411, 0.001518, 0.0003902, 0.0003902, 0.0003902,
#                      0.001221, 0.001221, 0.001221, 0.001221, 0.001221, 0.001221],
#     'K_ex':         [0.00157, 0.0002825, 0.0009723, 0.0002499, 0.0002499, 0.0002499, 
#                      0.0007822, 0.0007822, 0.0007822, 0.0007822, 0.0007822, 0.0007822],
#     'Mg_ex':        [0.02131, 0.003835, 0.0132, 0.003393, 0.003393, 0.003393, 
#                      0.01062, 0.01062, 0.01062, 0.01062, 0.01062, 0.01062],
#     'Na_ex':        [0.001596, 0.0002872, 0.0009888, 0.0002541, 0.0002541, 0.0002541,
#                      0.0007954, 0.0007954, 0.0007954, 0.0007954, 0.0007954, 0.0007954],
#     'Pyrite':       [0.0629, 0.0222, 0.0999, 0.01295, 0.01295, 0.01295, 
#                      0.13135, 0.13135, 0.13135, 0.13135, 0.13135, 0.13135]
    
# }

# # define icbund arrays for each layer for the transport model from the previous pht3d model
# layer_txt_files = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
#                    'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
#                    'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

# transport_ws = os.path.join(os.getcwd(), 'transport')

# pht3d_icbund = np.zeros((nlay, nrow, ncol))
# for i, layer_txt_files in enumerate(layer_txt_files):
#     pht3d_icbund[i] = reformat_arrays(os.path.join(transport_ws, 'props', 'icbund', layer_txt_files), 
#                                 os.path.join(transport_ws, 'props', 'icbund', 'format_'+str(layer_txt_files)), 
#                                 nrow, 
#                                 ncol)
    
# # define parameters for source-sink-mixing (SSM) package defined from SSM package file
# # specify six booleans in line 1 for types
# FWEL = 'T' # fwel flag is true
# FDRN = 'F' # fdrn flag is false
# FRCH = 'F' # frch flag is false
# FEVT = 'F' # fevt flag is false
# FRIV = 'F' # friv flag is false
# FGHB = 'F' # fghb flag is false

# ssm_line_1 = pd.DataFrame([[str(FWEL), str(FDRN), str(FRCH), str(FEVT), str(FRIV), str(FGHB)]])
# ssm_line_1.columns = [0, 1, 2, 3, 4, 5]  # Assuming the first column is the index column

# # define ssm line 2
# MXSS = 2245 # maximum number of source-sinks defined from SSM package file
# ssm_line_2 = pd.DataFrame([[MXSS]], columns=[0])

# # define all of the source-sink terms for the well boundary conditions with chemistry for all 29 constituents
# # lay, row, col, CSS (0=dummy value per manual), type (2=well), 
# # Orgc, O(0), C(4), C(-4), Ca, 
# # Cl, Fe(2), Fe(3), K, Mg, 
# # N(3), N(5), Na, S(-2), S(6), 
# # Si, Amm, N(0), Tmp, pH, 
# # pe, Ferrihydrite, Orgmatter, Ca_ex, Fe_ex, 
# # K_ex, Mg_ex, Na_ex, Pyrite

# # load in csv of ssm entries
# ssm_entry_rows = pd.read_csv(os.path.join(os.getcwd(), 'transport', 'ssm', 'ssm_entries.csv'), index_col=False)
# # Remove column headers from the CSV data
# ssm_entry_rows.columns = range(ssm_entry_rows.shape[1])  # Reset column names to default integer-based names

# # combine all entries to make ssm package dataframe

# ssm_combined = pd.concat([ssm_line_1, ssm_line_2, ssm_entry_rows], ignore_index=True, axis=0)

# # Check the first 5 columns for numeric values and round them to nearest integer for lay row col css type
# # We only modify the first 5 columns (0, 1, 2, 3, 4)
# for col in range(5):  # Loop through the first 5 columns
#     # Convert values to numeric, keeping 'T' and 'F' booleans intact
#     ssm_combined.iloc[1:, col] = pd.to_numeric(ssm_combined.iloc[1:, col], errors='coerce')

#     # Round numeric values to nearest whole number (integer), but leave 'T' and 'F' unaffected
#     ssm_combined.iloc[1:, col] = ssm_combined.iloc[1:, col].round()

# # print combined ssm 
# print(ssm_combined)
# # Now let's write the DataFrame to a text file while skipping NaN values
# file_path = os.path.join(os.getcwd(), 'transport', 'ssm', 'ssm_pkg_test.txt')  # Specify your file path
# # Convert DataFrame to string (without NaN values)
# ssm_combined_cleaned = ssm_combined.fillna('')  # Replace NaN with empty string if needed
# output_str = ssm_combined_cleaned.to_string(index=False, header=False)

# # Write the string to a text file
# with open(file_path, 'w') as file:
#     file.write(output_str)

# # import phinp.dat from pht3d model using phreeqc rm
# nxyz = nlay * ncol * nrow
# nthreads = 3

# phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz, nthreads)

# database_fpth = os.path.join(os.getcwd(), 'transport', 'database', 'dizon.pht3d_database')

# # Load the database
# phreeqc_rm.LoadDatabase(database_fpth)

# pqi_fpth = os.path.join(os.getcwd(), 'transport', 'pqi', 'phinp.dat')

# # Run the input file
# phreeqc_rm.RunFile(True, True, True, pqi_fpth)

def main():
    sim = make_gwf()
    sim = make_gwt(sim, model_name='Cl')
    run_model(sim)
if __name__ == "__main__":
    main()