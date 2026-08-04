import os
from pyexpat import features
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import pyemu
import geopandas as gpd
import platform
import shapefile as sf
from mf6rtm import utils, mup3d
from collections import defaultdict
from flopy.utils.gridintersect import GridIntersect
from collections.abc import Iterable
from pypestutils.pestutilslib import PestUtilsLib
lib = PestUtilsLib()

try:
    import keras_tuner as kt
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from tensorflow.keras.callbacks import EarlyStopping
    tf.config.run_functions_eagerly(True)
    mixed_precision.set_global_policy('mixed_float16')
    _HAS_TF = True
except ImportError:
    kt = tf = mixed_precision = EarlyStopping = None
    _HAS_TF = False  # surrogate path unavailable without tensorflow; model build does not need it
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture


import pickle

datadir = os.path.join("data")
dis_ws = os.path.join(datadir, 'dis')
props_ws = os.path.join(datadir, 'props')
# Kriging pilot-point values for the original ("og") K field, one row per layer per
# data/botm.shp feature. Extracted from the og model by export_og_pilot_props() so the
# structured build no longer needs a built model directory on disk -- see load_og_pilot_props.
OG_PROPS_CSV = os.path.join(props_ws, 'og_pilot_props.csv')

nper = 39  # Number of stress periods

perioddata= [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
            (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1), 
            (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1), 
            (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
            (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
            (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
            (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
            (28, 28, 1), (35, 35, 1), (35, 35, 1),(28, 28, 1)]

def get_avg_distance(points, npoints=10):
    """
    Calculate the average distance to the nearest n points for each point in a set of points.

    Parameters
    ----------
    points : numpy array
        Array of points.
    npoints : int
        Number of nearest points to calculate the average distance to.
    Returns
    -------
    average_distances : numpy array
        Array of average distances to the nearest n points for each point in the input array.
    """

    from scipy.spatial import distance
    distances = distance.cdist(points, points, 'euclidean')
    np.fill_diagonal(distances, np.inf)
    nearest_n = np.partition(distances, npoints, axis=1)[:, :npoints]
    average_distances = np.mean(nearest_n, axis=1)
    return average_distances

def interpolate_property_to_grid(gwf, ws, property_array, property_name="k"):
    """
    Interpolate a 3D property array (shape: nlay, nrow, ncol) onto the grid using kriging.
    """
    bps = gpd.read_file(os.path.join(datadir, 'botm.shp'))
    bps['x'] = bps.geometry.centroid.x
    bps['y'] = bps.geometry.centroid.y

    ppeasting = bps.x.values
    ppnorthing = bps.y.values
    anis = 1
    bearing = 0.0
    aa = 1.5 * get_avg_distance(bps[['x','y']].values, 10).max()

    easting = gwf.modelgrid.xcellcenters.flatten()
    northing = gwf.modelgrid.ycellcenters.flatten()
    max_pts = 50
    min_pts = 1
    search_dist = 1.e+10
    aa_pp = aa
    zone_pp = np.ones_like(ppeasting, dtype=int)
    fac_file = os.path.join(ws, f"factors.bin")

    ib = np.ones_like(easting, dtype=int)
    ipts = lib.calc_kriging_factors_2d(
        ppeasting, ppnorthing, zone_pp,
        easting, northing, ib.flatten(),
        "exp", "ordinary",
        aa_pp, anis, bearing, search_dist, max_pts, min_pts, fac_file, "binary"
    )

    nlay = gwf.dis.nlay.get_data()
    nrow = gwf.dis.nrow.get_data()
    ncol = gwf.dis.ncol.get_data()
    icpls = nrow * ncol

    interpolated = []
    for layer in range(nlay):
        # Flatten the property array for this layer
        ppval = property_array[layer].flatten()
        result = lib.krige_using_file(
            fac_file, "binary", icpls, "ordinary", "log",
            np.array(ppval), np.zeros_like(icpls), 0
        )
        # check if result['targval'] has negative values
        if np.any(result['targval'] < 0):
            print(f"Negative values found in result['targval'] for prop {property_name} layer {layer+1}")
        interpolated.append(np.round(result['targval'], 6))
    # check if inteprolated has negative values
    interpolated = np.array(interpolated).reshape((nlay, nrow, ncol))
    if np.any(interpolated < 0):
        print(f"Negative values found in interpolated array for prop {property_name}")
        raise ValueError("Interpolated values contain negative values.")
    return np.array(interpolated)


def _botm_pilot_points():
    """(x, y) of the data/botm.shp centroids -- the kriging pilot points, in file order."""
    bps = gpd.read_file(os.path.join(datadir, 'botm.shp'))
    return bps.geometry.centroid.x.values, bps.geometry.centroid.y.values


def export_og_pilot_props(ws=os.path.join("model", "reactive_demo"), out=OG_PROPS_CSV):
    """One-off: extract the og K pilot-point values from a built model into OG_PROPS_CSV.

    The og model is a 12x10x51 grid and data/botm.shp has exactly 510 features, one per og cell,
    so each layer of `npf.k`/`npf.k33` is really a vector of pilot-point values keyed to that
    shapefile -- no grid semantics are needed to reuse them. Kept in the repo so the CSV's origin
    is documented and regenerable rather than a mystery blob.
    """
    sim_og = flopy.mf6.MFSimulation.load(sim_ws=ws, sim_name='gwf', version='mf6',
                                        exe_name='mf6', verbosity_level=0,
                                        load_only=['dis', 'npf'])
    gwf_og = sim_og.get_model("gwf")
    ppx, ppy = _botm_pilot_points()
    rows = []
    for p in ('k', 'k33'):
        arr = np.asarray(getattr(gwf_og.npf, p).get_data())
        nlay = arr.shape[0]
        for lay in range(nlay):
            vals = arr[lay].flatten()
            assert len(vals) == len(ppx), (
                f"{p} layer {lay + 1} has {len(vals)} cells but botm.shp has {len(ppx)} "
                "features -- the pilot-point mapping no longer holds")
            for i, v in enumerate(vals):
                rows.append((lay + 1, i, ppx[i], ppy[i], p, float(v)))
    df = pd.DataFrame(rows, columns=['layer', 'pp_index', 'x', 'y', 'prop', 'value'])
    df = df.pivot_table(index=['layer', 'pp_index', 'x', 'y'], columns='prop',
                        values='value').reset_index()
    df.columns.name = None
    df = df[['layer', 'pp_index', 'x', 'y', 'k', 'k33']].sort_values(['layer', 'pp_index'])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows ({df.layer.nunique()} layers x {df.pp_index.nunique()} pts)")
    return df


def load_og_pilot_props(csv=OG_PROPS_CSV, atol=1e-6):
    """{'k': (nlay, npp), 'k33': (nlay, npp)} pilot-point values for the og K field.

    Verifies the stored x/y against the live data/botm.shp centroids, so a reordered or edited
    shapefile fails loudly instead of silently pairing values with the wrong locations -- the one
    way this indirection could corrupt the K field without anything looking wrong.
    """
    df = pd.read_csv(csv).sort_values(['layer', 'pp_index'])
    ppx, ppy = _botm_pilot_points()
    npp = len(ppx)
    lay0 = df[df.layer == df.layer.min()]
    if len(lay0) != npp:
        raise ValueError(f"{csv} has {len(lay0)} pilot points per layer but data/botm.shp has {npp}")
    if not (np.allclose(lay0.x.values, ppx, atol=atol)
            and np.allclose(lay0.y.values, ppy, atol=atol)):
        raise ValueError(f"{csv} pilot-point coordinates do not match data/botm.shp centroids; "
                         "regenerate it with export_og_pilot_props()")
    nlay = df.layer.nunique()
    return {p: df[p].values.reshape((nlay, npp)) for p in ('k', 'k33')}


def get_properties(gwf, fac_ws, prop=('k', 'k33')):
    """Krige the og K pilot-point values onto `gwf`'s grid; returns {prop: (nlay,nrow,ncol)}.

    Replaces the former get_properties_from_og, which loaded model/reactive_demo -- a 3.6 GB
    gitignored artifact that prep_model_dir wipes, so a fresh clone could not build at all. The
    values now come from OG_PROPS_CSV and `fac_ws` (the directory being built) receives the
    kriging scratch file, instead of it being written back into the og model directory.

    Only k and k33 are kriged. `ss` in the og model is a uniform 1e-3, and kriging a constant
    field returns that constant, so it is set directly. `sy` was never passed to ModflowGwfsto
    (iconvert=0, the aquifer is confined), so kriging it was pure waste.
    """
    pilot = load_og_pilot_props()
    nlay = gwf.dis.nlay.get_data()
    nrow = gwf.dis.nrow.get_data()
    ncol = gwf.dis.ncol.get_data()
    dict_prop = {p: interpolate_property_to_grid(gwf, fac_ws, pilot[p], p) for p in prop}
    # full array rather than a scalar so set_all_data_external() writes the same external file
    dict_prop['ss'] = np.full((nlay, nrow, ncol), 1.e-3)
    return dict_prop

def get_botms(gwf, ws):
    bps = gpd.read_file(os.path.join(datadir, 'botm.shp'))
    bps['x'] = bps.geometry.centroid.x
    bps['y'] = bps.geometry.centroid.y
    bps

    ppeasting = bps.x.values
    ppnorthing = bps.y.values
    anis = 1
    bearing= 0.0
    aa = 1.5 * get_avg_distance(bps[['x','y']].values, 2).max()

    ib = gwf.dis.idomain.get_data()
    # cellids = df.loc[df.layer==layer+1].icpl.values - 1 # zero-based
    easting = gwf.modelgrid.xcellcenters.flatten()
    northing = gwf.modelgrid.ycellcenters.flatten()

    max_pts = 50 # pp are same as cell centers, so kind of irrelevant
    min_pts = 1
    search_dist = 1.e+10
    aa_pp = aa #?
    zone_pp = np.ones_like(ppeasting,dtype=int)
    fac_file = os.path.join(ws,f"factors.bin")

    ib = np.ones_like(easting,dtype=int)
    ipts = lib.calc_kriging_factors_2d(ppeasting,
                                    ppnorthing,
                                    zone_pp,
                                    easting,
                                    northing,
                                    ib.flatten(),
                                    "exp","ordinary",
                                    aa_pp,anis,bearing,search_dist,max_pts,min_pts,fac_file,"binary")

    botms = []
    icpls = gwf.dis.nrow.get_data() * gwf.dis.ncol.get_data()
    for layer in range(1, 13):
        # get COND multiplier
        ppval = bps[f"botm_{layer}"].values
        result = lib.krige_using_file(os.path.join(ws,f"factors.bin"),
                                        "binary",
                                        icpls,
                                        "ordinary",
                                        "none",
                                        np.array(ppval),
                                        np.zeros_like(icpls),
                                        0)
        botms.append(np.round(result['targval'], 1))
    return botms

def make_gwf_structured(ws, model_name = "gwf", tracer = 'Cl', mup3d_m = None, write = True):

    nper = 39  # Number of stress periods

    perioddata= [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
                (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1), 
                (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1), 
                (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (35, 35, 1), (35, 35, 1),(28, 28, 1)]

    # specify the mf6 gw object & add relevant components
    sim = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', sim_ws='.')
    sim.set_sim_path(ws)

    # specify tdis
    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS", 
                                 nper=nper, perioddata=perioddata)
    outer_dvclose = 1e-7
    inner_dvclose = 1e-7
    ims = flopy.mf6.ModflowIms(sim, 
                            #    pname="ims", 
                            complexity="complex",
                            outer_dvclose=outer_dvclose,
                            inner_dvclose=inner_dvclose,
                            filename=f"{model_name}.ims")
    sim.register_ims_package(ims, 
                             [model_name])
    
    # start model build to refine 
    model_nam_file = "{}.nam".format(model_name)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, 
                               model_nam_file=model_nam_file, exe_name='mf6')

    length_units = "METERS"

    domain = gpd.read_file(os.path.join(datadir, 'modeldis.shp'))
    Ly = domain.geometry.total_bounds[3] - domain.geometry.total_bounds[1]
    Lx = domain.geometry.total_bounds[2] - domain.geometry.total_bounds[0]

    xul = domain.geometry.total_bounds[0]
    yul = domain.geometry.total_bounds[1]

    dis = make_dis_structured(gwf, Ly=Lx, Lx=Ly, xul=xul, yul=yul)
    botms = get_botms(gwf, ws)
    dis.botm.set_data(botms)
    dis.set_all_data_external()

    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()

    # dis.idomain.export(os.path.join("output", "dis_botm.vtk"), fmt="vtk")
    dict_prop = get_properties(gwf, fac_ws=ws, prop=('k', 'k33'))

    ihead = 0 #(meters)
    strt = ihead * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)
    ic.set_all_data_external()

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=dict_prop['k'],
        k33=dict_prop['k33'],
        # k33overk = True
    )
    npf.set_all_data_external()
    # DEAD: unused -- sto below takes ss from dict_prop (1e-3), not this 1e-4. Left in place
    # because it looks like an unrealised intent to use 1e-4; changing it would alter the flow
    # field, so it needs a deliberate decision rather than a silent refactor.
    ss = np.ones((nlay, nrow, ncol)) * 1.e-4
    sto = flopy.mf6.ModflowGwfsto(gwf,
                                  ss=dict_prop['ss'],
                                  iconvert=0,
                                    # steady_state={0: False},
                                    transient={0: True})
    sto.set_all_data_external()

    chd = make_chd(gwf, one_compound=tracer, mup3d_m=mup3d_m)

    wel_out = make_wel_out(gwf, nper =39, one_compound=tracer, mup3d_m=mup3d_m, from_shp=True)
    # # make wel in
    wel_in = make_wel_in(gwf, nper =39, 
                        one_compound=tracer, mup3d_m=mup3d_m, from_shp=True)

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
    if write:
        sim.write_simulation()
    return sim


def make_dis_structured(gwf, Ly=10, Lx=10, xul=0, yul=0):
    # Define model parameters for the dummy model
    del_ = 8  # Grid spacing
    nrow = int(Lx / del_)+1
    ncol = int(Ly / del_)+1
    nlay = 12  # Number of layers
    ib = np.ones((nlay, nrow, ncol))

    top = -273  # Top elevation (constant, meters above mean sea level)
    botm = np.zeros((nlay, nrow, ncol))  # Bottom elevations of each layer (meters)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=del_,
        delc=del_,
        top=top,
        idomain=ib,
        botm=botm,
        xorigin=xul,
        yorigin=yul
    )
    return dis
# def copy_gwt_model_files_from_parent(ws=".", parent_model_dir = 'gwtbenzene',
#                                      dsp_par =  ['alh', 'ath1', 'atv'], 
#                                  ist_par = ['porosity', 'zetaim', 'volfrac', 'bulk_density'],
#                                  mst_par = ['buk_density']
#                                  ):
#     import shutil
#     def get_transport_input_filenames(tag, template_ws=os.path.join('pest','pst_template'), 
#                                     gwtdir = 'gwtbenzene'):
#         template_ws = os.path.join(template_ws, gwtdir)
#         files = [os.path.join(gwtdir,f) for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")] 
#         return files 
#     def flatten(xss):
#         return [x for xs in xss for x in xs]
#     tag = []
#     modeldirs = [name for name in os.listdir(ws) if (os.path.isdir(os.path.join(ws, name)) ) and (name.startswith('gwt'))]

#     assert parent_model_dir in modeldirs, f'{parent_model_dir} does not exist in dir of models'
    
#     modeldirs.remove(parent_model_dir)

#     for e, par in enumerate(dsp_par):
#         tag.append(f"dsp_{par}_")

#     for  e, par in enumerate(ist_par):
#          tag.append(f"ist_{par}_")

#     for  e, par in enumerate(mst_par):
#          tag.append(f"mst_{par}_")
         
#     fnames_to_copy = sorted([get_transport_input_filenames(t, template_ws=ws, gwtdir = parent_model_dir) for t in tag])
#     fnames_to_copy = flatten(fnames_to_copy)
    
#     for dir in modeldirs:
#         fnames_to_replace = sorted([get_transport_input_filenames(t, template_ws=ws, gwtdir = dir) for t in tag])
#         fnames_to_replace = flatten(fnames_to_replace)
#         assert sorted([x.split('.')[1] for x in fnames_to_copy]) == sorted([x.split('.')[1] for x in fnames_to_replace]), f'list of files to replace and to copy does not contain the same files names '
#         # sort fnames_to_copy and fnames_to_replace according to the assert above
#         fnames_to_copy = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_copy], fnames_to_copy))]
#         fnames_to_replace = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_replace], fnames_to_replace))]
#         fileszipped = list(zip(fnames_to_copy, fnames_to_replace))
#         [shutil.copyfile(os.path.join(ws, f[0]), os.path.join(ws, f[1])) for f in fileszipped]
#     return fileszipped

def clean_obs_chem(datadir = "data",
                    input_path="obs_chem_raw_0.csv", 
                   output_path="obs_chem_cleaned.csv"):
    """Load, clean, and save obs_chem data by converting units of specific variables."""
    input_path = os.path.join(datadir, input_path)
    output_path = os.path.join(datadir, output_path)
    obsdata = pd.read_csv(input_path)
    obsdata.loc[:, 'obsid'] = obsdata['obsid'].str.lower()

    data_source2 = pd.read_csv(os.path.join("data", "obs_chem_raw_1.csv"))
    to_pull = ['Fe2', 'Fe3', 'tic']
    data_source2=data_source2[data_source2["var"].isin(to_pull)].copy()
    data_source2['units'] = 'mol_l'
    data_source2=data_source2[obsdata.columns]

    obsdata = pd.concat([obsdata, data_source2])

    # Standardize column name
    obsdata.rename(columns={'var': 'variable'}, inplace=True)
    
    # Unit conversions
    unit_conversions = {
        # 'Cl': 1/(58.44*1e3),   # mmol/L → mol/L
        'Tmp': 1e-3,  # apply scaling to be consistent with outputs
    }

    for var, factor in unit_conversions.items():
        obsdata.loc[obsdata['variable'] == var, 'value'] *= factor

    # Save cleaned file
    obsdata.to_csv(output_path, index=False, float_format="%.5e")

    return obsdata

def time_interpolate(sim_times, sim_vals, obs_times):
    import numpy as np
    from scipy import interpolate
    t0 = min(sim_times)
    sim_times = [(i-t0).astype(float) for i in sim_times]
    obs_times = [(i-t0).astype(float) for i in obs_times]

    # Create interpolation function
    f = interpolate.interp1d(sim_times, sim_vals, fill_value='extrapolate')
    # Interpolate at new times
    new_values = f(obs_times)
    return new_values

def create_output_pairs(perioddata, output_interval=5):
    pairs = []
    cumulative_day = 0
    next_output_day = 0
    
    for kper, (perlen, nstp, tsmult) in enumerate(perioddata):
        period_days = int(perlen)
        
        # Check each day in this stress period
        for day_in_period in range(period_days):
            if cumulative_day == next_output_day:
                pairs.append((kper+1, day_in_period+1))
                next_output_day += output_interval
            
            cumulative_day += 1
    
    return pairs

def append_values_to_inner_lists(d, values, *, in_place=False):
    """
    Append a single value or all values from an iterable to every inner list
    inside a {key: list[list]} dictionary.

    Parameters
    ----------
    d : dict
        Your nested list dictionary.
    values : any or Iterable
        * If `values` is not an Iterable (or is str/bytes), its treated as a
          single item and appended once.
        * If `values` is an Iterable (list/tuple/set/range), each element is
          appended in order.
    in_place : bool, default False
        True  → modify `d` directly and return it.  
        False → leave `d` unchanged and return a *new* dictionary.

    Returns
    -------
    dict
        The dictionary with updated inner lists.
    """
    # Decide whether to work on the original or a shallow copy
    target = d if in_place else {k: [lst[:] for lst in v] for k, v in d.items()}

    is_iterable = (
        isinstance(values, Iterable) and
        not isinstance(values, (str, bytes))  # treat strings/bytes as scalars
    )

    for outer in target.values():
        for inner in outer:
            if is_iterable:
                inner.extend(values)   # add every element in order
            else:
                inner.append(values)   # add the single value
    return target

def run_model(sim):
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

def make_obs_pack(gwf):
    ix = GridIntersect(gwf.modelgrid)
    obsloc = pd.read_csv(os.path.join(datadir, "obs_loc.csv"))

    obs_list=[]

    for obsid in obsloc.obsid.unique():
        x,y = obsloc.loc[obsloc.obsid==obsid,['x','y']].values[0]
        cellid = ix.intersect([(x,y)],shapetype="point").cellids
        if len(cellid)==0:
            print(f"{obsid} not in model domain")
            continue
        else:
            cellid = cellid[0]
            print(f"{obsid} is in model domain") 
        obs_layer = int(obsloc.loc[obsloc.obsid==obsid,'layer'].values[0])
        obs_list.append((obsid, 'concentration', (obs_layer, cellid[0], cellid[1])))

    # obs_recarray = {'obs.head.sim.csv':obs_list}
    obs_recarray = {f'obs_{gwf.name}.csv':obs_list}
    # print(obs_list)
    obs_package = flopy.mf6.ModflowUtlobs(gwf, 
                                        digits=0, #print_input=True,
                                        pname=f'obs_{gwf.name}',
                                        continuous=obs_recarray)
    return obs_package

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
        xorigin=-148,
        yorigin=0
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

def initialize_chemistry(ws, nlay, nrow, ncol, sim=None, gwt_name='Cl'):
    # get chemistry

    solutionsdf = pd.read_csv(os.path.join(datadir,"ic_aq_chem.csv"), index_col = 0)

    ## let's process the injection chem
    injdf = pd.read_csv(os.path.join(datadir,"wellin.csv"), index_col = 0)
    injdf = injdf[['layer'] + solutionsdf.index.tolist()].copy()

    frames = []                     

    for per in injdf.index.unique():
        df = (
            injdf.loc[per].reset_index()      
                .drop(columns='kper')        
                .groupby('layer').mean()    
                .T                          
        )
        # give columns names like  0_1, 0_2, …   (or f"{per}_{layer}")
        df.columns = [f"{per}_{layer}" for layer in df.columns]

        frames.append(df)            

    injdf = pd.concat(frames, axis=1) 
    solutionsdf = pd.concat([solutionsdf, injdf], axis=1)

    solutions = utils.solution_df_to_dict(solutionsdf)

    sol_ic = np.ones((nlay, nrow, ncol), dtype=float)

    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)
    solution

    excdf = pd.read_csv(os.path.join(os.path.join(datadir,"ic_exchanger.csv")), comment = '#')
    ex_names = {
                "Ca_ex": "CaX2",
                "Fe_ex":"FeX2" ,
                "K_ex":"KX" ,
                "Mg_ex":"MgX2", 
                "Na_ex":"NaX"
                }
    # we need to rename to match the database
    excdf['name'] = excdf['var'].map(ex_names)
    excdf['layer'] -= 1  # convert to zero‑indexed
    excdf = excdf.pivot(index="name", columns="layer", values="value")
    exchanger_dict = excdf.to_dict()
    for k, subdict in exchanger_dict.items():
        for key in subdict:
            subdict[key] = {'m0': subdict[key]}

    exchanger = mup3d.ExchangePhases(exchanger_dict)

    layer_vals = np.arange(1, nlay + 1, dtype=float)   # shape (nlay,)

    # Broadcast to full 3‑D grid
    exchanger_ic = layer_vals[:, None, None] * np.ones((1, nrow, ncol))
    exchanger.set_ic(exchanger_ic)

    # we need to equilibrate the exchangers with the background solution
    # init background solution is the same (number 1) for the whole domain
    # the next array has a length of nlay, one for each layer
    eq_solutions = [1] * nlay
    exchanger.set_equilibrate_solutions(eq_solutions)
    
    mindf = pd.read_csv(os.path.join(os.path.join(datadir,"ic_surfaces.csv")), comment = '#')
    mindf['value'] = [utils.concentration_volbulk_to_volwater(i,0.35) for i in mindf['value'].values]
    mindf = mindf.pivot(index="var", columns="layer", values="value")
    
    # only ferrihydrite and orgmatter are in eq
    eq_m0 = utils.solution_df_to_dict(mindf.loc[['Ferrihydrite', "Orgmatter"],:])

    # we need the initial Sat indeces SI
    # following original model init SI is 0
    si = 0

    # Ferrihydrite as an equilibrium phase: Prommer & Stuyfzand (2005) docs/est0486768.pdf
    # p. 2202 include "mineral equilibrium for ferrihydrite (Fe(OH)3)" in their reaction
    # network, and Fe(OH)3 is the product of BOTH pyrite oxidation reactions, so the reference
    # model precipitates the Fe(3) that pyrite oxidation releases. With it off, Fe(3) has no
    # mineral sink and stays dissolved at ~2.5e-4 mol/L at pH 6.8 -- far above ferrihydrite
    # solubility -- which is the leading explanation for the WP1 pH offset vs PHT3D.
    # m0 = 0 in every layer of ic_surfaces.csv, so it acts as a precipitate-only sink.
    #
    # Ferrihydrite was absent from every pre-2026 deck only because of a bug: main's eq_dic loop
    # re-initialised `eq_dic[ly] = {key: {}}` *inside* the per-mineral loop, so each mineral wiped
    # the previous one and only the last (Orgmatter) survived. a42fec6 fixed that. The N and redox
    # parameters were therefore fitted while this sink was accidentally missing -- enabling it
    # improves the NO3/SO4 fit slightly but degrades pH (RMSE 0.158 -> 0.194), so a recalibration
    # is outstanding.
    # NOTE: must be matched in the PHT3D twin (pht3d_species_csv type-D rows) or the two codes
    # are no longer comparable.
    eq_keys = ["Ferrihydrite", "Orgmatter"]

    eq_dic = {}
    for ly in range(nlay):
        eq_dic[ly] = {}
        for key in eq_keys:
            # si followed by m0 (init moles) for each equilibrium mineral in this layer
            eq_dic[ly][key] = {'si': si, 'm0': eq_m0[key][ly]}
    equilibriums = mup3d.EquilibriumPhases(eq_dic)
    equilibriums.set_ic(exchanger_ic)

    # pyrite is kinetic
    py_m0 = utils.solution_df_to_dict(mindf.loc[["Pyrite"],:])

    # we need the kinetic params for Py
    kin_py_params = [1.600000e+01, 
                     6.700000e-01, 
                     5.000000e-01, 
                     -1.100000e-01]
    kin_dic = {}

    # lets add kinetics for Organic carbon (Orgc)
    kin_orgc_params = [1.570000e-09, 1.670000e-11, 1.000000e-13]
    # orgc also have a custom formula
    orgc_form =  "Orgc -1.0 CH2O 1.0"
    orgc_steps = "8.640000e+04 in 1 steps"

    #lets add pyrite first
    for ly in range(nlay):
        # print(ly+1)
        for key in py_m0.keys():
            kin_dic[ly] = {key: {}}
            kin_dic[ly][key]['m0'] = py_m0[key][ly]
            kin_dic[ly][key]['parms'] = kin_py_params
            # kin_dic[ly+1] = {key: [py_m0[key][ly], kin_py_params]}

    # lets add orgc now with a m0 of zero for all layers
    for key in kin_dic.keys():
        kin_dic[key]['Orgc'] = {}
        kin_dic[key]['Orgc']['m0'] = 1.0
        kin_dic[key]['Orgc']['parms'] = kin_orgc_params
        kin_dic[key]['Orgc']['formula'] = orgc_form
        kin_dic[key]['Orgc']['steps'] = orgc_steps
        # [1.0, kin_orgc_params, orgc_form, orgc_steps]
    kinetics = mup3d.KineticPhases(kin_dic)
    kinetics.set_ic(exchanger_ic)
    if sim is not None:
        # from_mf6: flopy builds GWF + a conservative Cl tracer GWT; mup3d clones the
        # tracer GWT into one reactive GWT per PHREEQC component at write_simulation()
        model = mup3d.Mup3d.from_mf6(sim, solution, name='dizon36', gwt_name=gwt_name)
    else:
        model = mup3d.Mup3d('dizon36',solution, nlay, nrow, ncol)

    # #set model workspace
    model.set_wd(ws)

    # set database
    # shutil copy datab
    shutil.copy(os.path.join(datadir, f'datab.dat'), os.path.join(model.wd, f'datab.dat'))
    database = os.path.join(f'datab.dat')
    model.set_database(database)

    postfix = os.path.join(datadir, f'postfix.phqr')
    model.set_postfix(postfix)
    model.set_exchange_phases(exchanger)
    model.set_phases(kinetics)
    model.set_phases(equilibriums)
    # model.set_charge_offset(1e-3)
    tsteps = create_output_pairs(perioddata, output_interval=2)

    # machine learning variables
    targetvars = [
    'Orgc','O0','tic','C_4',
    'Fe2','Fe3','N3','NO3',
    'S_2','SO4','Amm',
    'N0','pH',
    'pe','EQUI_Ferrihydrite',
    'EQUI_Orgmatter',
    'MOL_CaX2','MOL_FeX2',
    'MOL_KX','MOL_MgX2',
    'MOL_NaX','KIN_Pyrite'
    ]
    featvars = [
    'Orgc',
     'O0', #NOTE: including all, because they are needed to calculate the change
     'tic',
     'C_4',
     'Fe2',
     'Fe3',
     'N3',
     'NO3',
     'S_2',
     'SO4',
     'Amm',
     'N0',
     'pH',
     'pe',
    'EQUI_Ferrihydrite',
    'EQUI_Orgmatter',
    'MOL_CaX2','MOL_FeX2',
    'MOL_KX','MOL_MgX2',
    'MOL_NaX','KIN_Pyrite'
    ]

    model.set_config(
                    reactive_timing='all',
                    reactive_externalio=True,
                    solver_threshold=0.0,   # react every cell every step (0.3.2 'epsaqu'=0);
                                            # 0.5.1's default 1e-10 skips near-static cells and
                                            # freezes kinetic phases -> wrong Fe/redox
                    emulator_training_data=True,
                    emulator_target_variables=targetvars,
                    emulator_feature_variables=featvars,
                    # tsteps=tsteps
                    )
    model.set_componenth2o(True)
    model.initialize(add_charge_flag=True)
    return model

def get_wel_coords(gwf, name  = "wellin"):
    mg = gwf.modelgrid
    ix = GridIntersect(mg)
    wells = pd.read_csv(os.path.join(datadir, "wells.csv"))
    wells = gpd.GeoDataFrame(wells, geometry=gpd.points_from_xy(wells.x, wells.y))

    assert name in wells.name.values, f"{name} not in well"
    wells = wells[wells.name == name]
    geom = wells.geometry[wells.name==name].values

    assert len(geom)==1, f"more than one well with name {name} in wells.csv"
    cellid = ix.intersect(geom[0], 'point').cellids
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    mg.plot(ax=ax)
    ix.plot_point(ix.intersect(geom[0], 'point'), ax=ax)
    print(cellid[0])
    return cellid[0]

def make_wel_in(gwf, one_compound = None,
                mup3d_m=None, nper=39, from_shp=False):
    layers = [1,2,3,5,7]
    if from_shp:
        # coords_in = {}
        cellid = get_wel_coords(gwf, name  = "wellin")
        coords_in = {lay: (lay, cellid[0], cellid[1]) for lay in layers}

        print(cellid)
    else:
        coords_in = [
            (1, 9, 38),
            (2, 9, 38),
            (3, 9, 38),
            (5, 9, 38),
            (7, 9, 38),
        ]
    df_inj = pd.read_csv(os.path.join(datadir, "wellin.csv"))
    wellin_sp_data = defaultdict(list)

    if one_compound is not None:
        assert one_compound in df_inj.columns, print("compound not in wellin csv")
        #get all unique cells
        for _, r in df_inj.iterrows():
            if from_shp:
                layer = int(r["layer"])
                cell = coords_in[layer]  # zero‑indexed
            else:
                cell = (int(r["layer"]), int(r["row"]), int(r["column"]))  # zero‑indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"], r[f"{one_compound}"]])
        wel_in  = flopy.mf6.ModflowGwfwel(gwf, 
                                       stress_period_data=wellin_sp_data,
                                       auxiliary=one_compound,
                                       pname = 'welin',
                                       filename=f'{gwf.name}.welin')
        wel_in.set_all_data_external()

    else:
        wel_chem_dir = {}
        indices = [list(range(i, i + 5)) for i in range(2, 197, 5)]
        for per in range(nper):
            sol_spd = indices[per]
            wellchem = mup3d.ChemStress('per_'+str(per))
            wellchem.set_spd(sol_spd)
            mup3d_m.set_chem_stress(wellchem)
            wel_chem_dir[per] = wellchem.data

        for _, r in df_inj.iterrows():
            if from_shp:
                layer = int(r["layer"])
                cell = coords_in[layer]  # zero‑indexed
            else:
                cell = (int(r["layer"]), int(r["row"]), int(r["column"]))  # zero‑indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"]])

        for per in range(nper):
            for e, layer in  enumerate(layers):
                chem_arr = wel_chem_dir[per][e]
                wellin_sp_data[per][e].extend(chem_arr)
        wel_in  = flopy.mf6.ModflowGwfwel(gwf, 
                                        stress_period_data=wellin_sp_data,
                                        auxiliary=mup3d_m.components,
                                        pname = 'welin',
                                        filename=f'{gwf.name}.welin')
        wel_in.set_all_data_external()
        return wel_in


def make_wel_out(gwf, one_compound = None, mup3d_m=None, nper=39, from_shp=False):

    if from_shp:
        layers = [1,3,5]
        cellid = get_wel_coords(gwf, name  = "wellout")
        coords_out = [(lay, cellid[0], cellid[1]) for lay in layers]
        print(coords_out)
    else:
        coords_out = [ # (row, col, layer) 
            (1, 9,  9),
            (3, 9,  9),
            (5, 9,  9),
        ]
    init_rates_out  = [-300,  -30,  -30]                # 3 negatives
    fini_rates_out  = [-400,  -40,  -40]

    init_sp = range(0, 35)   # stress periods 0 – 35
    fini_sp = range(35, nper)  # stress periods 36 – 38
    all_sp  = (*init_sp, *fini_sp)


    def make_rows(coords, rates, add_conc=True):
        if len(coords) != len(rates):
            raise ValueError("Coordinate and rate lists must be the same length")
        return [
            ([cell, q] if add_conc else [cell, q])
            for cell, q in zip(coords, rates)
        ]

    # Time‑invariant blocks for each phase
    if from_shp:
        wellout_init = {sp: make_rows(coords_out, init_rates_out) for sp in all_sp}
        wellout_fini = {sp: make_rows(coords_out, fini_rates_out, add_conc=True) 
                        for sp in all_sp}
        wellout_sp_data = {sp: (wellout_init[sp] if sp in init_sp else wellout_fini[sp])
                    for sp in all_sp}
    else:
        wellout_init  = make_rows(coords_out, init_rates_out)
        wellout_fini  = make_rows(coords_out, fini_rates_out, add_conc=True)
        wellout_sp_data = {sp: (wellout_init if sp in init_sp else wellout_fini)
                    for sp in all_sp}

    if one_compound is not None:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, 0.0)
        wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                            stress_period_data=wellout_sp_data, 
                                            auxiliary=one_compound,
                                            pname = 'welout' ,
                                            filename=f'{gwf.name}.welout')
        wel_out.set_all_data_external()
    else:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, [0.0]*len(mup3d_m.components))
        wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                            stress_period_data=wellout_sp_data, 
                                            auxiliary=mup3d_m.components,
                                            pname = 'welout',
                                            filename=f'{gwf.name}.welout')
        
        wel_out.set_all_data_external()
    return wel_out

def make_chd(gwf, one_compound = None, mup3d_m=None):
    
    l_hd= 0
    nlay = gwf.dis.nlay.get_data()
    nrow = gwf.dis.nrow.get_data()
    ncol = gwf.dis.ncol.get_data()

    if one_compound is not None:
        df_inj = pd.read_csv(os.path.join(datadir, "ic_aq_chem.csv"), index_col=0)
        assert one_compound in df_inj.index, f"compound {one_compound} not in ic_aq_chem csv"
        c_list = [df_inj.loc[one_compound, 'value']]
        aux = one_compound
        # print(c_list)
    else:
        chdchem = mup3d.ChemStress('chdchem')
        sol_spd = [1]
        chdchem.set_spd(sol_spd)
        mup3d_m.set_chem_stress(chdchem)
        c_list = mup3d_m.chdchem.data[0]
        aux=mup3d_m.components

    chdspd = []
    for i in range(nlay):          # layers
        for j in range(nrow):      # rows
            chdspd.append([(i, j, 0), l_hd])           # left boundary
            chdspd.append([(i, j, ncol - 1), l_hd])    # right boundary
    # print(chdspd)
    for i in range(len(chdspd)):
        chdspd[i].extend(c_list)

    # print(chdspd)
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=True,
        auxiliary=aux,
        pname="CHD",
        filename=f"{gwf.name}.chd",
    )
    chd.set_all_data_external()
    return chdspd

def prep_model_dir(name="model"):
    model_ws = os.path.join("model", name)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)

    # Re‑create an empty folder
    os.makedirs(model_ws, exist_ok=True)
    utils.prep_bins(model_ws)
    return os.path.join(model_ws)

def make_gwf(ws, model_name = "gwf", tracer = 'Cl', mup3d_m = None):

    nper = 39  # Number of stress periods

    perioddata= [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
                (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1), 
                (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1), 
                (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (35, 35, 1), (35, 35, 1),(28, 28, 1)]

    # specify the mf6 gw object & add relevant components
    sim = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', sim_ws='.')
    sim.set_sim_path(ws)

    # specify tdis
    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS", 
                                 nper=nper, perioddata=perioddata)
    outer_dvclose = 1e-7
    inner_dvclose = 1e-7
    ims = flopy.mf6.ModflowIms(sim, 
                            #    pname="ims", 
                            complexity="complex",
                            outer_dvclose=outer_dvclose,
                            inner_dvclose=inner_dvclose,
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
    # k = np.zeros((nlay, nrow, ncol))
    # k33 = np.zeros((nlay, nrow, ncol))
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
        # k33overk = True
    )
    npf.set_all_data_external()

    sto = flopy.mf6.ModflowGwfsto(gwf, 
                                  ss=scoeff96/thick96, 
                                  iconvert=0,
                                    # steady_state={0: False},
                                    transient={0: True})
    sto.set_all_data_external()

    chd = make_chd(gwf, one_compound=tracer, mup3d_m=mup3d_m)

    wel_out = make_wel_out(gwf, nper =39, one_compound=tracer, mup3d_m=mup3d_m)
    # make wel in
    wel_in = make_wel_in(gwf, nper =39, one_compound=tracer, mup3d_m=mup3d_m)

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

    return sim

def make_gwt(sim, tracer = 'Cl', mup3d_m=None, write=True):

    gwf = sim.get_model("gwf")
    nlay = gwf.dis.nlay.get_data()

    ne = 0.35 # effective porosity (-) constant across all layers
    long_disp = 0.1 # Longitudinal dispersivity (m)constant across all layers
    disp_tr_vert = long_disp*0.01 # Transverse vertical dispersivity (m) constant across all layers
    disp_tr_hor = long_disp*0.1 # Transverse horizontal dispersivity (m) constant across all layers
    diffc = 0 # diffusion coefficient constant across all layers
    pbulk = 1850 # bulk density in constant across all layers (m/L^3) 

    if mup3d_m is not None and tracer is None:
        components = mup3d_m.components
    else:
        components = [tracer]

    for comp in components:
        print(f"Setting transport for {comp}")
        # For the conservative tracer template (tracer is not None, used by from_mf6), name the
        # model 'tracer' so it does not collide with the 'Cl' PHREEQC component that from_mf6
        # clones. The well aux/SSM still reference `comp` (='Cl'), so the tracer follows Cl.
        model_name = 'tracer' if tracer is not None else comp
        gwt = flopy.mf6.MFModel(
            sim,
            model_type="gwt6",
            modelname=model_name,
            model_nam_file=f"{model_name}.nam"
        )
        outer_dvclose = 1e-7
        inner_dvclose = 1e-7
        ims = flopy.mf6.ModflowIms(sim, 
                                #    pname="ims", 
                                complexity="complex",
                                outer_dvclose=outer_dvclose,
                                inner_dvclose=inner_dvclose,
                                filename=f"{model_name}.ims")
        sim.register_ims_package(ims, 
                                [model_name])

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
                xorigin=gwf.dis.xorigin.get_data(),
                yorigin=gwf.dis.yorigin.get_data()
            )
        dis.set_all_data_external()

        nlay = dis.nlay.get_data()
        nrow = dis.nrow.get_data()
        ncol = dis.ncol.get_data()
        if mup3d_m is None:
            strt = 0.0  # conservative tracer template for from_mf6; background handled by PHREEQC solutions
        elif tracer is not None:
            strt = mup3d_m.sconc[comp]/1000 # to mmol
        else:
            strt = mup3d_m.sconc[comp]
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

        sourcerecarray = [
                        ["welin", "aux", comp],
                        ["welout", "aux", comp],
                        ["chd", "aux", comp]
                        ]

        ssm = flopy.mf6.ModflowGwtssm(
                gwt,
                sources=sourcerecarray,
                save_flows=True,
                print_flows=True,
                filename=f"{model_name}.ssm",
            )
        ssm.set_all_data_external()

        if comp=='Tmp':
            # kd = 2*ne/1850
            distcoef = np.ones(shape=(nlay,ncol, nrow))*2.1141E-04
            sorption = "Linear"
            pbulk = 1850
            bulk_density = np.ones(shape=(nlay,ncol, nrow))*pbulk
            ne=0.35
        else:
            distcoef = None
            sorption = None
            bulk_density = None
            ne=0.35
        porosity  = np.ones(shape=(nlay,ncol, nrow))*ne
        
        mst = flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=porosity,
            first_order_decay=None,
            decay = None,
            decay_sorbed=None,
            sorption= sorption,
            bulk_density=bulk_density, 
            distcoef=distcoef, #Kd m3/mg
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
        make_obs_pack(gwt)

    if write:
        sim.write_simulation()
    return sim

def clean_array_files(ws, files, gwf):
    nrows, ncols = gwf.dis.nrow.get_data(), gwf.dis.ncol.get_data()
    for file in files:
        with open(os.path.join(ws,file),'r') as f:
            lines = []
            for l in f.readlines():
                lines.extend([float(i) for i in l.split()])
        array = np.array(lines).reshape(nrows,ncols)
        np.savetxt(os.path.join(ws,file),array,fmt='%.6e')
    return

# def get_lst_budget(ws='.',start_datetime=None):
#     import flopy
#     lst = flopy.utils.Mf6ListBudget(os.path.join(ws,"pl253.lst"))
#     inc,cum = lst.get_dataframes(diff=True,start_datetime=start_datetime)
#     inc.columns = inc.columns.map(lambda x: x.lower().replace("_","-"))
#     cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
#     inc.index.name = "time"
#     cum.index.name = "time"
#     inc.to_csv(os.path.join(ws,"inc.csv"))
#     cum.to_csv(os.path.join(ws,"cum.csv"))
#     return inc, cum

def tidy_array_input_files(f, template_ws=os.path.join('pest','pst_template')):
    """Re-writes array input files as a single column."""
    filename=os.path.join(template_ws, f)
    with open(filename, 'r') as ff:
        a = ff.read()
    a = [float(i) for i in a.split()]
    # make a single column
    a = np.array(a).reshape(-1,1)
    # record array to txt file in single column
    np.savetxt(os.path.join(template_ws, f), a, fmt='%.8e')
    return

def get_input_filenames(tag, template_ws=os.path.join('pest','pst_template')):
    files = [f for f in os.listdir(template_ws) if tag in f and f.endswith(".txt")] 
    return files 

def copy_obs_data_files_to_wd(datadir,wd,files=[]):
    for f in files:
        assert os.path.isfile(os.path.join(datadir,f)),f"file {f} not found"
        print(f"Getting {f} from {datadir} to {wd}")
        shutil.copy(os.path.join(datadir,f), os.path.join(wd, f))

def draw_prior_pe(num_reals, pf, template_ws):
    sigma_range=4.0
    if pf.pst.npar < 36000:  #if you have more than about 35K pars, the cov matrix becomes hard to handle
        prior_cov = pf.build_prior(fmt='coo',
                                   filename=os.path.join(template_ws,"prior_cov.jcb"),
                                   sigma_range=sigma_range)
        pf.pst.pestpp_options["parcov"] = "prior_cov.jcb"
        # draw ensemble
        pe = pyemu.ParameterEnsemble.from_gaussian_draw(pf.pst, cov=prior_cov, num_reals=num_reals)
    else:
        # draw parameters from the prior distribution
        pe = pf.draw(num_reals=num_reals, use_specsim=False, sigma_range=sigma_range) 

    ### Fix params in boundary TVMs
    params = list(pe.columns)
    to_fix = [i for i in params if ('boundary' in i) & ('tvm' in i)]
    pe.loc[:, to_fix] = 1
    # enforces parameter bounds
    pe.enforce() 
    # save external file
    pe.to_binary(os.path.join(template_ws,"prior_pe.jcb")) 
    pf.pst.pestpp_options["ies_parameter_ensemble"] = "prior_pe.jcb"
    return pe

def process_sim_conc(wd='.'):
    from flopy.utils.gridintersect import GridIntersect

    sim = flopy.mf6.MFSimulation.load(sim_ws = wd,
                                    sim_name = 'gwf', 
                                    version='mf6',
                                        exe_name='mf6',
                                        verbosity_level=0)
    gwf = sim.get_model("gwf")
    sout = pd.read_csv(os.path.join(wd, "sout.csv"))

    # nlay = gwf.dis.nlay.get_data()
    ncol = gwf.dis.ncol.get_data()
    nrow = gwf.dis.nrow.get_data()

    ix = GridIntersect(gwf.modelgrid)
    obsloc = pd.read_csv(os.path.join(wd, "obs_loc.csv"))

    obs_list=[]

    for obsid in obsloc.obsid.unique():
        x,y = obsloc.loc[obsloc.obsid==obsid,['x','y']].values[0]
        cellid = ix.intersect([(x,y)],shapetype="point").cellids
        if len(cellid)==0:
            continue
        else:
            cellid = cellid[0]
            obs_layer = int(obsloc.loc[obsloc.obsid==obsid,'layer'].values[0])
            obs_list.append((obsid, 'concentration', (obs_layer, cellid[0], cellid[1])))

    wells = pd.DataFrame(obs_list)
    wells.columns = ['obsid', 'n', 'cell']
    wells['flat_index'] = [cell[0] * (nrow * ncol) + cell[1] * ncol + cell[2] for cell in wells.cell]

    obs_to_index = dict(zip(wells['flat_index'], wells['obsid'].str.lower()))

    sout['cell'] = sout['cell'].astype(int)
    sout['obsid'] = sout['cell'].map(obs_to_index)

    obsdata = pd.read_csv(os.path.join(wd, "obs_chem_cleaned.csv"))
    obsdata.rename(columns={'var': 'variable'},  inplace=True)

    missvar = set(obsdata['variable'].unique()) - set(sout.columns)
    obs_ = sout[['time', 'cell']+list(set(obsdata['variable'].unique())  - missvar)].copy()
    obs_ = obs_.melt(id_vars = ['time', 'cell'])
    obs_['obsid'] = obs_['cell'].map(obs_to_index)
    obs_['obsid'].unique()

    obs_in_ = obs_[~obs_['obsid'].isna()].copy()
    variables = obs_in_.variable.unique()
    obsids = obs_in_.obsid.unique()

    obsdata = obsdata[obsdata['obsid'].isin(obsids)].copy()
    obsdata = obsdata[obsdata['variable'].isin(variables)].copy()

    dfmerged = pd.merge(obs_in_[['time','obsid', 'variable', 'value']], obsdata,
                        on=['time','obsid', 'variable'], how='outer')
    dfmerged.rename(columns={'value_x':'sim',
                            'value_y': 'meas'}, inplace=True)

    dfmerged.sort_values(['obsid','time'], inplace=True)
    dfmerged.set_index('time', inplace=True)

    for oid in obsids:
        print(f"Processing obs for: {oid:>5}")
        for var in variables:
            mask=(dfmerged.obsid==oid)&(dfmerged.variable==var)

            tmp = dfmerged.loc[mask].copy()
            tmp.dropna(subset=['sim'], inplace=True)
            if tmp.shape[0]==0:
                continue
            obs_times = dfmerged.loc[mask].index.values
            dfmerged.loc[mask,'sim'] = time_interpolate(tmp.index.values,
                                                        tmp.sim.values,
                                                        obs_times)

    fname = '_obs.conc.simvsmeas.csv'
    dfmerged = dfmerged.reset_index()
    dfmerged.drop_duplicates(subset=['time', 'obsid', 'variable'], inplace=True)
    dfmerged = dfmerged.set_index('time')
    dfmerged.replace(np.nan,1e30).to_csv(os.path.join(wd, fname))
    print(f"Processed conc saved in {wd}/{fname}")

    return fname

def add_std_to_pst(casename="dizon36",
                    template_ws=os.path.join("pest","pst_template"),
                    fraction=1):
    pst = pyemu.Pst(os.path.join(template_ws,f"{casename}.pst"))
    obs = pst.observation_data
    obs['standard_deviation'] = obs['obsval']*fraction
    pst.write(os.path.join(template_ws, f"{casename}.pst"), version=2)

def build_noise_ensemble(casename="dizon36",
                         template_ws=os.path.join("pest","pst_template")):

    pst = pyemu.Pst(os.path.join(template_ws,f"{casename}.pst"))
    obs = pst.observation_data
    cov = pyemu.Cov.from_observation_data(pst)
    # get number of reals 
    pefname = os.path.join(template_ws, pst.pestpp_options["ies_parameter_ensemble"])
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=pefname)
    num_reals = pe.shape[0]

    # gerenate noise ensemble
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals)

    assert oe.columns.isin(obs.obsnme).all()

    #---save ensemble---#
    fname = os.path.join(template_ws, 'noise.jcb')
    oe.to_binary(fname)
    print(f'saving noise ensemble to: {fname}')

    pst.pestpp_options["ies_observation_ensemble"] = 'noise.jcb'
    pst.write(os.path.join(template_ws, f"{casename}.pst"), version=2)
    return oe

def setup_pest(org_d, num_reals=50, 
               template_ws=os.path.join('pest','pst_template'),
               casename = 'dizon36'):
    tmp_d=os.path.join("tmp_d")
    if os.path.exists(tmp_d):
        shutil.rmtree(tmp_d)
    shutil.copytree(org_d,tmp_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d, verbosity_level=0)
    # load flow model
    gwf = sim.get_model()
    # get spatial reference
    modelgrid = gwf.modelgrid
    # get zone array
    ib = gwf.dis.idomain.get_data()
    ib[ib<1] = 0

    pf = pyemu.utils.PstFrom(original_d=tmp_d, 
                                new_d=template_ws,
                                remove_existing=True, 
                                longnames=True, 
                                spatial_reference=modelgrid, 
                                zero_based=False, 
                                # start_datetime=start_datetime, 
                                echo=False)

    pf.mod_sys_cmds.append("mf6rtm")
    pf.extra_py_imports.append("flopy")
    pf.add_py_function("workflow.py","process_sim_conc()",is_pre_cmd=False)
    pf.add_py_function("workflow.py","time_interpolate()",is_pre_cmd=None)

    copy_obs_data_files_to_wd(datadir,template_ws,files=['obs_chem_cleaned.csv', 'obs_loc.csv'])
    f = process_sim_conc(wd=template_ws)
    obs_df = pf.add_observations(f, 
                            insfile=f+".ins", 
                            index_cols=['time','obsid','variable'], 
                            use_cols=['sim'], 
                            prefix=f"hm") 
    pp_v = pyemu.geostats.ExpVario(contribution=1,
                                   a=75,
                                   anisotropy=4,
                                   bearing=90.0)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v, transform='log')

    tag_list = ["npf_k_", 
                # "npf_k33_",
                "sto_ss_",
                "kinetic_phases.Pyrite.m0.",
                # "equilibrium_phases.Orgmatter.m0.",
                # "exchange_phases.CaX2.m0",
                # "exchange_phases.FeX2.m0",
                # "exchange_phases.KX.m0",
                # "exchange_phases.MgX2.m0",
                # "exchange_phases.NaX.m0",
                ]
    lb=0.1
    ub=10.0
    for tag in tag_list:
    # uub=100.0
    # ulb=0.01
        files = get_input_filenames(tag, template_ws=template_ws)
        clean_array_files(template_ws, files, gwf)
        for f in files:
            try:
                # layer = int(f.split(".")[1].split("_layer")[-1]) -1
                layer = int(f.split(tag)[1].split('.txt')[0].split("layer")[-1])
            except:
                layer=0
            # print(layer)
            base = tag.replace("_",".")+'layer'+str(layer)
            print(base)
            pf.add_parameters(filenames=f,
                                par_type="pilotpoints",
                                par_name_base='pp.'+base,
                                pargp='pp.'+base,
                                # zone_array=ib[layer],
                                # use_pp_zones=True,
                                upper_bound=ub,
                                lower_bound=lb,
                                # ult_ubound=uub,
                                # ult_lbound=ulb,
                                pp_options={"pp_space":2,
                                            "prep_hyperpars":False},
                                geostruct=pp_gs,
                                # apply_order=2
                                )
            pf.add_parameters(f, 
                                # zone_array=ib[layer],
                                par_type="constant",
                                # geostruct=pp_gs,
                                par_name_base="cn."+base,
                                par_style='m',
                                pargp="cn."+base,
                                lower_bound=lb,
                                upper_bound=ub,
                                # ult_ubound=uub,
                                # ult_lbound=ulb
                                )

            pf.add_observations(f,
                                prefix=base,
                                obsgp=base,
                                # zone_array=ib[layer]
                                )

    pst = pf.build_pst()
    pe = draw_prior_pe(num_reals, pf, template_ws)
    if pf.pst.npar < 35000:
        pst.pestpp_options["parcov"] = "prior_cov.jcb"
    pst.pestpp_options["ies_parameter_ensemble"] = "prior_pe.jcb"
    pst.write(os.path.join(template_ws, f'{casename}.pst'), version=2)


def run_pestpp(md=f"master", td="pst_template", casename="isr", 
               noptmax=-1,freeze=False,
               num_reals=100,
               num_workers=10, worker_root=".", 
               pestpp_version="ies",restart=False,
               reuse_master=False, cleanup=True):

    pst = pyemu.Pst(os.path.join(td, f"{casename}.pst"))
    pst.control_data.noptmax = noptmax

    # and a options to reduce the number of lost reals and improve conditioning
    pst.pestpp_options["overdue_giveup_fac"] = 1e30
    pst.pestpp_options["overdue_giveup_minutes"] = 1e30
    pst.pestpp_options["ies_no_noise"] = False 
    pst.pestpp_options["ies_subset_size"] = -10 # the more the merrier
    pst.pestpp_options["ies_bad_phi_sigma"] = 2.0
    pst.pestpp_options["panther_agent_freeze_on_fail"] = freeze
    pst.pestpp_options["ies_num_reals"] = num_reals

    pst.write(os.path.join(td, f"{casename}.pst"), version=2)
    # run
    pyemu.os_utils.start_workers(td,
                                 f'pestpp-{pestpp_version}',
                                 f'{casename}.pst',
                                 num_workers=num_workers,
                                 worker_root=worker_root,
                                 master_dir=md,
                                 reuse_master=reuse_master,
                                 )
    return

def set_obsval_and_weights(casename="dizon36",
                           template_ws=os.path.join("pest","pst_template")):
    
    '''Set observation parval1 and weights for history matching in pst
    '''
    pst = pyemu.Pst(os.path.join(template_ws,f"{casename}.pst"))

    obs = pst.observation_data
    obs.weight = 0.0

    hm_obs = obs.oname == 'hm'
    zero_weight_obs = obs.loc[obs.obsval>=1e30].obsnme
    zero_weight_obs

    obs_hm = obs.loc[hm_obs].copy()
    obs_hm.sort_values(['obsid','time'], inplace=True)
    obs_hm['time'] = obs_hm['time'].astype(float)

    obs_chem = pd.read_csv(os.path.join(template_ws, "_obs.conc.simvsmeas.csv"))
    obs_chem.sort_values(['obsid','time'], inplace=True)
    obs_chem['variable'] = obs_chem['variable'].str.lower()

    assert obs_hm.shape[0] == obs_chem.shape[0]

    obs_chem = pd.merge(obs_hm, obs_chem, on=['time','obsid', 'variable'])
    obs_chem.loc[obs_chem.meas<1e30, 'weight'] = 1.0
    obs_chem['obgnme'] = obs_chem['variable']
    obs.loc[obs_chem.obsnme, 'obsval'] = obs_chem.meas.values
    obs.loc[obs_chem.obsnme, 'weight'] = obs_chem.weight.values
    assert obs.loc[(obs.oname=='hm') & (obs.weight>0)].shape[0] == obs_chem.loc[obs_chem.meas<1e30].shape[0]
    assert obs.loc[(obs.oname=='hm') & (obs.weight>0)].weight.sum() == obs_chem.loc[obs_chem.meas<1e30].shape[0]
    obs.loc[zero_weight_obs, 'weight'] = 0.0
    obs.loc[obs_chem.obsnme, 'obgnme'] = obs_chem.obgnme.values #oname per var

    pst.write(os.path.join(template_ws, f"{casename}.pst"), version=2)
    return pst


def get_trainingdata():
    ws = os.path.join("model","reactive")

    features = pd.read_csv(os.path.join(ws, '_features.csv'))
    features.sort_values(by=['time','cell'], inplace=True)
    #features.rename(columns={'time':'t0'}, inplace=True)

    targets = pd.read_csv(os.path.join(ws, '_targets.csv'))
    targets.sort_values(by=['time','cell'], inplace=True)
    #targets.rename(columns={'time':'t1'}, inplace=True)

    df = pd.concat([features, targets], axis=1, keys=['features','targets'])

    return df

def preprocess_data(X, y,test_size=0.2):

    #X = X.drop(columns=['time','cell'])
    #y = y.drop(columns=['time','cell'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale the features
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    # scale the output features
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return X_train, X_test, X_scaler, y_train, y_test, y_scaler



class LogStandardScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler: log-transform + StandardScaler.

    - Applies log10(x + eps) to selected columns before scaling.
    - Applies only StandardScaler to all columns.
    - Inverse transform can return both linear and log-transformed values.

    Parameters
    ----------
    log_cols : list of str
        Columns to log-transform (must be strictly positive).
    eps : float, default=1e-12
        Small value to avoid log(0).
    """
    def __init__(self, log_cols=None, eps=1e-12):
        self.log_cols = log_cols if log_cols is not None else []
        self.eps = eps
        self.scaler = StandardScaler()
        self.col_order = None
        self.shifts = {}  # Store shift per column

    def fit(self, X, y=None):
        X = X.copy()
        self.shifts = {}
        for col in self.log_cols:
            min_val = X[col].min()
            shift = 0.0
            if min_val <= 0:
                shift = abs(min_val) + self.eps
            self.shifts[col] = shift
            X[col] = np.log10(X[col] + shift)
        self.col_order = X.columns.tolist()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.log_cols:
            shift = self.shifts.get(col, 0.0)
            X[col] = np.log10(X[col] + shift)
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.col_order)
        return X_scaled

    def inverse_transform(self, X_scaled, return_log=False):
        """
        Inverse-transform scaled data.

        Parameters
        ----------
        X_scaled : np.ndarray
            Scaled data (as from transform).
        return_log : bool, default=False
            If True, also return the log-transformed values (before exp).

        Returns
        -------
        pd.DataFrame
            DataFrame with either linear concentrations (default)
            or both linear and log values (if return_log=True).
        """
        # Back to log-space
        X_log = self.scaler.inverse_transform(X_scaled)
        X_log = pd.DataFrame(X_log, columns=self.col_order)

        # Copy for output
        X_out = X_log.copy()

        # Inverse log-transform only on selected cols
        for col in self.log_cols:
            shift = self.shifts.get(col, 0.0)
            X_out[col] = np.power(10, X_log[col]) - shift

        if return_log:
            # Add extra columns with ".log" suffix
            for col in self.log_cols:
                X_out[f"{col}.log"] = X_log[col]

        return X_out

    def _apply_log(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.col_order if self.col_order else None)
        for col in self.log_cols:
            shift = self.shifts.get(col, self.eps)
            X[col] = np.log10(X[col] + shift)
        return X



class ClusterLogScaler(BaseEstimator, TransformerMixin):
    """
    Preprocessor combining:
    - Log10(+shift) + StandardScaler for selected cols
    - Cluster-aware residual scaling for multimodal cols

    Parameters
    ----------
    cluster_config : dict {col_name: n_clusters}
        Number of clusters per feature (1 = continuous).
    log_cols : list of str
        Columns to apply log10 transform before scaling.
    eps : float
        Small positive shift for log safety.
    """

    def __init__(self, cluster_config, log_cols=None, eps=1e-12):
        self.cluster_config = cluster_config
        self.log_cols = log_cols if log_cols is not None else []
        self.eps = eps

        # fitted objects
        self.shifts = {}        # shifts for log cols
        self.gmms = {}          # fitted GMMs
        self.scalers = {}       # scalers per col (or per cluster)
        self.col_order = None   # to reconstruct inverse transform
        self.max_points = 100000

    def fit(self, X, y=None):
        X = X.copy()
        self.shifts = {}
        self.scalers = {}

        # Loop over all columns
        for col in X.columns:
            x = X[col].values.reshape(-1, 1)
            


            # --- apply log shift if needed ---
            if col in self.log_cols:
                print(f"Applying log shift for {col}")
                min_val = x.min()
                shift = 0.0
                if min_val <= 0:
                    shift = abs(min_val) + self.eps
                self.shifts[col] = shift
                x = np.log10(x + shift)

            # --- cluster-aware scaling ---
            if col in self.cluster_config.keys():
                print(f"Fitting GMM for {col}")
                n_clusters = self.cluster_config[col]

                if n_clusters > 1:

                    if self.max_points and len(x) > self.max_points:
                        idx = np.random.choice(len(x), self.max_points, replace=False)
                        x_fit = x[idx].reshape(-1, 1)
                    else:
                        x_fit = x

                    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=42)
                    gmm.fit(x_fit)
                    self.gmms[col] = gmm

                    cluster_assignments = gmm.predict(x)

                    # only keep clusters with enough samples
                    min_samples = 5
                    valid_clusters = [
                        c for c in range(n_clusters)
                        if np.sum(cluster_assignments == c) >= min_samples
                    ]
                    self.scalers[col] = {
                        c: StandardScaler().fit(x[cluster_assignments == c])
                        for c in valid_clusters
                    }
                    self.scalers[col][col] = StandardScaler().fit(x)  # fallback scaler
            else:
                print(f"Fitting StandardScaler for {col}")
                self.scalers[col] = StandardScaler().fit(x)

        self.col_order = list(X.columns.tolist())
        return self

    def transform(self, X):
        X = X.copy()
        X_out = pd.DataFrame(columns=X.columns)

        for col in X.columns:
            x = X[col].values.reshape(-1, 1)

            # --- log transform if needed ---
            if col in self.log_cols:
                shift = self.shifts.get(col, 0.0)
                x = np.log10(x + shift)

            if col in self.cluster_config.keys():
                n_clusters = self.cluster_config[col]
                if n_clusters > 1:
                    gmm = self.gmms[col]
                    cluster_assignments = gmm.predict(x)

                    # one-hot encode clusters
                    cluster_ids = pd.get_dummies(cluster_assignments, prefix=f"{col}_cluster")
                    cluster_ids = cluster_ids.astype(int)
                    for i in range(n_clusters):
                        cname = f"{col}_cluster_{i}"
                        if cname not in cluster_ids:
                            cluster_ids[cname] = 0
                    X_out = pd.concat([X_out, cluster_ids], axis=1)

                    # residuals
                    residuals = np.zeros_like(x, dtype=float)
                    for c in range(n_clusters):
                        idx = cluster_assignments == c
                        if np.any(idx):
                            residuals[idx] = self.scalers[col][c].transform(x[idx])
                    X_out[f"{col}_resid"] = residuals.flatten()

                    X_out[col] = self.scalers[col][col].transform(x).flatten()

            else:
                X_out[col] = self.scalers[col].transform(x).flatten()

        return X_out

    def inverse_transform(self, X_proc):
        X_recon = pd.DataFrame(index=X_proc.index)

        for col, n_clusters in self.cluster_config.items():
            if n_clusters > 1:
                cluster_cols = [c for c in X_proc.columns if c.startswith(f"{col}_cluster")]
                cluster_ids = X_proc[cluster_cols].values.argmax(axis=1)
                residuals = X_proc[f"{col}_resid"].values.reshape(-1, 1)

                x_recon = np.zeros_like(residuals)
                for c in range(n_clusters):
                    idx = cluster_ids == c
                    if np.any(idx):
                        x_recon[idx] = self.scalers[col][c].inverse_transform(residuals[idx])
                x = x_recon

            else:
                x = self.scalers[col].inverse_transform(X_proc[[col]].values)

            # inverse log if applied
            if col in self.log_cols:
                shift = self.shifts.get(col, 0.0)
                x = np.power(10, x) - shift

            X_recon[col] = x.flatten()

        return X_recon



def build_neuralnet(hp,input_shape,output_shape,loss_fn):
    model = tf.keras.Sequential()
    # First hidden layer
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units1', min_value=32, max_value=2*256, step=32),
        activation='relu',
        input_shape=(input_shape,)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Float('dropout1', 0.0, 0.5, step=0.1)))
    # Second hidden layer
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units2', min_value=32, max_value=2*256, step=32),
        activation='relu'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Float('dropout2', 0.0, 0.5, step=0.1)))
    # Third hidden layer
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units3', min_value=32, max_value=2*256, step=32),
        activation='relu'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(hp.Float('dropout3', 0.0, 0.5, step=0.1)))
    # Output layer
    model.add(tf.keras.layers.Dense(output_shape))
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('lr', 1e-5, 1e-2, sampling='log')
        ),
        loss=loss_fn,
        metrics=['mae','mse']
    )
    return model

def instantiate_tuner(input_shape, output_shape, loss_fn):
    tuner = kt.BayesianOptimization(
        lambda hp: build_neuralnet(hp, input_shape=input_shape, output_shape=output_shape, loss_fn=loss_fn),
        objective='val_loss',
        max_trials=20,
        directory='tuning_dir',
        project_name='nnet_tuning',
    )
    return tuner

def hyper_parameter_tuning(X, y, sample_size, tuner):
    # Randomly sample a subset of the data for tuning
    idxs = np.random.choice(np.arange(X.shape[0]), size=sample_size, replace=False)
    X = X.sample(n=sample_size, random_state=42)
    assert X.shape[0] == sample_size
    y = y.loc[X.index]
    assert y.shape[0] == sample_size
    # assert no nans
    # get the sum of null values
    badcols = X.isnull().sum().sort_values()
    
    assert not X.isnull().values.any(), X[badcols[badcols>0].index]
    assert not y.isnull().values.any()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    tuner.search(
        X_train, y_train,
        validation_data=(X_test,y_test),
        epochs=100,
        batch_size=256 * 16,
        callbacks=[early_stop]
    )
    
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model



# create a surrogate Class
class SurrogateModel:

    def __init__(self, model, scalers):
        """Initialize the surrogate model with a trained model and scalers.

        model: A trained machine learning model.
        scalers: A dictionary of scalers for feature and target variables.
        """
        self.model = model
        self.scaler = scalers

    def predict(self, X, apply_scale_to_X=True, batch_size=256):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            try:
                X = X.drop(columns=['time','cell'])
            except KeyError:
                pass
        if apply_scale_to_X:
            if "X" in self.scaler.keys() and self.scaler["X"] is not None:
                X = self.scaler["X"].transform(X)
        pred = self.model.predict(X,batch_size=batch_size)
        if "y" in self.scaler.keys():
            pred = self.scaler["y"].inverse_transform(pred)
        return pred
    
    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    def load(model_path):
        # instantiate an empty SurrogateModel class and load attributes form pickle
        surrogate = SurrogateModel.__new__(SurrogateModel)
        with open(model_path, 'rb') as f:
            surrogate.__dict__ = pickle.load(f).__dict__
        return surrogate



def plot_y_vs_yhat(columns,result_dict):

    #save to pdf
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    #y = y.drop(columns=['time','cell'])
    #yhat = pd.DataFrame(yhat, columns=y.columns)

    marker_dict = {
        "train":{"color":'0.5', "marker":'o', "size": 20},
        "test":{"color":'b', "marker":'x', "size": 10}
    }

    with PdfPages('y_vs_yhat.pdf') as pdf:
        for e,col in enumerate(columns):
            print(col)

            fig,axs = plt.subplots(1,2,figsize=(8,4),sharex=True,sharey=True)

            for e,key in enumerate(result_dict.keys()):
                ax= axs[e]
                y = result_dict[key]['y'].loc[:,col]
                yhat = result_dict[key]['yhat'].loc[:,col]
                #ax.scatter(y, yhat, label=key,
                #           alpha=0.3,
                #           c=marker_dict[key]['color'],
                #           marker=marker_dict[key]['marker'])
                hx = ax.hexbin(y, yhat, gridsize=200,mincnt=1, cmap='viridis',bins='log')
                cb = fig.colorbar(hx, ax=ax, shrink=0.5,label='log10(count)')

                ax.set_title(f"{col} - {key}")
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                #ax.legend()

                # set y and x limits to same values
                lims = ax.get_xlim()
                lims2 = ax.get_ylim()
                xmax = max(lims[1], lims2[1])
                xmin = min(lims[0], lims2[0])
                ax.plot([xmin, xmax], [xmin, xmax], 'k--')
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([xmin, xmax])
                
                ax.set_aspect('equal')
            fig.tight_layout()
            pdf.savefig(fig,dpi=90)
            #plt.show()

            

    return

_LossBase = tf.keras.losses.Loss if _HAS_TF else object
class PhysicsInformedLoss(_LossBase):
    """
    Physics-informed custom loss for geochemical surrogate models.

    Penalizes:
        1. Standard MSE between predicted and true values
        2. Negative concentrations (non-negativity)
        3. Mass imbalance (element totals vs species)
        4. Charge imbalance
        5. Redox inconsistency across electron-transfer couples

    Parameters
    ----------
    cols : list of str
        Column names in the same order as model outputs.
    element_balances : dict
        Mapping {element: [species_list]} for mass balance.
    redox_couples : dict
        Mapping {name: (oxidized_species, reduced_species, n_electrons)}.
    weights : dict
        Optional weights for each penalty component, e.g.:
        {
            "mse": 1.0, "nonneg": 0.1, "mass": 0.5,
            "charge": 0.5, "redox": 0.3
        }
    charges : dict
        Ionic charges of species for charge balance.
    eps : float
        Small number to prevent log(0) or division by zero.
    """

    def __init__(self, 
                 cols,
                 non_zero_columns,
                 element_balances,
                 redox_couples,
                 charges,
                 scaler,
                 weights=None,
                 eps=1e-12,
                 name="PhysicsInformedLoss"):
        super().__init__(name=name)
        self.cols = cols
        self.non_zero_columns = non_zero_columns
        self.non_zero_columns_idxs = [self.cols.index(col) for col in non_zero_columns]
        self.element_balances = element_balances
        self.redox_couples = redox_couples
        self.charges = charges
        self.eps = eps
        self.scaler = scaler

        default_weights = {"mse": 1.0, "nonneg": 0.1, "mass": 0.5, "charge": 0.5, "redox": 0.3}
        self.weights = weights if weights is not None else default_weights

    # -------------------
    # Mass balance penalty
    # -------------------
    def mass_balance_penalty(self, y_pred):
        penalties = []
        for elem, species_list in self.element_balances.items():
            if elem not in self.cols:
                continue
            total = y_pred[:, self.cols.index(elem)]
            summed_species = 0.0
            for sp in species_list:
                if sp in self.cols:
                    summed_species += y_pred[:, self.cols.index(sp)]
            penalties.append(tf.reduce_mean(tf.square(total - summed_species)))
        return tf.add_n(penalties) if penalties else 0.0

    # -------------------
    # Charge balance penalty
    # -------------------
    def charge_balance_penalty(self, y_pred):
        total_charge = 0.0
        for sp, z in self.charges.items():
            if sp in self.cols:
                total_charge += z * y_pred[:, self.cols.index(sp)]
        if "Charge" in self.cols:
            predicted_charge = y_pred[:, self.cols.index("Charge")]
            return tf.reduce_mean(tf.square(predicted_charge - total_charge))
        else:
            return 0.0

    # -------------------
    # Redox penalty
    # -------------------
    def redox_penalty(self, y_pred):
        penalties = []
        for name, (ox, red, n) in self.redox_couples.items():
            if all(sp in self.cols for sp in [ox, red, "pe"]):
                ox_val = y_pred[:, self.cols.index(ox)] + self.eps
                red_val = y_pred[:, self.cols.index(red)] + self.eps
                pe_implied = (1.0 / n) * tf.math.log(ox_val / red_val) / tf.math.log(10.0)
                penalties.append(tf.reduce_mean(
                    tf.square(y_pred[:, self.cols.index("pe")] - pe_implied)
                ))
        return tf.add_n(penalties) if penalties else 0.0

    # -------------------
    # Call method
    # -------------------
    def call(self, y_true, y_pred):

        # MSE in transformed space
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # physics based metrics in original space; inverse transform first
        y_pred_orig = self.scaler.inverse_transform(y_pred.numpy())
        
        # assert no nans
        assert y_pred_orig.isnull().sum().sum() == 0, "NaNs in inverse transformed predictions"
        nonneg = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred_orig.iloc[:, self.non_zero_columns_idxs])))
        assert not np.isnan(nonneg.numpy()), "NaNs in non-negativity penalty"
        mass = self.mass_balance_penalty(y_pred_orig.values)
        #assert not np.isnan(mass.numpy()),mass.numpy()# "NaNs in mass balance penalty"
        charge = self.charge_balance_penalty(y_pred_orig.values)
        #assert not np.isnan(charge.numpy()), "NaNs in charge balance penalty"

        #TODO: something wrong with redox function
        #redox = self.redox_penalty(y_pred_orig)
        #assert not np.isnan(redox.numpy()), "NaNs in redox penalty"

        total_loss = (self.weights["mse"] * mse +
                      self.weights["nonneg"] * nonneg +
                      self.weights["mass"] * mass +
                      self.weights["charge"] * charge 
                      #+ self.weights["redox"] * redox
                      )
        #assert not np.isnan(total_loss.numpy()), "NaNs in total loss"
        return total_loss



def surrogate_workflow(hyperparameter_tuning=False):

    # data etl
    data_fpath = os.path.join(".",'dataframe.pkl')
    if os.path.exists(data_fpath):
        df = pd.read_pickle(data_fpath)
    else:
        df = get_trainingdata()
        df.to_pickle( os.path.join(".",'dataframe.pkl') )
    
    X = df['features'].drop(columns=['time','cell'])
    y = df['targets'].drop(columns=['time','cell'])



    # split and fit scaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # specify log cols
    log_cols = X.columns.tolist()
    remove_cols = ["ph","tmp","pe","sturation","KIN_","EQUI_","MOL_"]
    
    log_cols = X.loc[:,~X.columns.str.lower().str.contains('|'.join(remove_cols))].columns.tolist()
    for c in ["N","Fe","N","O0","C_4","Fe2","Fe3","NO3","N0"]:
        try:
            log_cols.remove(c)
        except ValueError:
            pass
    #for c in remove_cols:
    #    log_cols = log_cols[~log_cols.str.lower().str.contains(c)]
    #log_cols = X.columns[(X > 0).all(axis=0)].tolist()
    #X_scaler = LogStandardScaler(log_cols=log_cols)
    cluster_config = {
                    "pe": 5,
                    "EQUI_Orgmatter": 5,
                    "MOL_CaX2": 5,
                    "MOL_FeX2": 5,
                    "MOL_KX": 5,
                    "MOL_MgX2": 5,
                    "KIN_Pyrite": 5,
                }
    X_scaler = ClusterLogScaler(log_cols=log_cols,
                                cluster_config=cluster_config
                                )
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    #log_cols = y.columns[(y > 0).all(axis=0)].tolist()
    y_scaler = LogStandardScaler(log_cols=log_cols)
    #y_scaler = ClusterLogScaler(log_cols=log_cols,
    #                            cluster_config=cluster_config
    #                            )
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # Columns from your DataFrame
    cols = y.columns.tolist()

    # Element balances
    ELEMENT_BALANCES = {
        "C": ["tic", "Orgc", "Orgc.1", "C_4"],
        "N": ["NO3", "N3", "N0", "Amm"],
        "S": ["SO4", "S_2"],
        "Fe": ["Fe2", "Fe3", "MOL_FeX2"],
        "O": ["H2O", "O0", "NO3", "SO4"]
    }

    # Redox couples
    REDOX_COUPLES = {
        "Fe": ("Fe3", "Fe2", 1),
        "N": ("NO3", "N0", 5),
        "S": ("SO4", "S_2", 8)
    }

    # Charges
    CHARGES = {
        "H": +1, "Ca": +2, "Cl": -1, "K": +1, "Mg": +2, "Na": +1,
        "Fe2": +2, "Fe3": +3, "NO3": -1, "N3": -3, "Amm": +1,
        "S_2": -2, "SO4": -2
    }

    # get all column names that do not contain <= 0
    non_zero_columns = X.columns[(X > 0).any(axis=0)]

    # Instantiate loss
    loss_fn = PhysicsInformedLoss(
        cols=cols,
        non_zero_columns=non_zero_columns,
        scaler=y_scaler,
        element_balances=ELEMENT_BALANCES,
        redox_couples=REDOX_COUPLES,
        charges=CHARGES   
    )
    
    
    
    loss_fn = "mse"

    tuner = instantiate_tuner(input_shape=X_train.shape[1],
                              output_shape=y_train.shape[1],
                              loss_fn=loss_fn)
    if hyperparameter_tuning:
        m = hyper_parameter_tuning(X_train,
                                y_train,
                                sample_size=100000,
                                tuner=tuner
                                )
    else:
        m = tuner.get_best_models(num_models=1)[0]

    
    
    m.fit(X_train, y_train,
          #loss_fn=loss_fn,
          epochs=100,
          validation_data=(X_test, y_test),
          batch_size=256*16,
          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
          verbose=1)
    
    s = SurrogateModel(m, {"X": X_scaler, "y": y_scaler})

    s.save('surrogate_model.pkl')
    batch_size = 2048 * 4
    yhat_train = s.predict(X_train, apply_scale_to_X=False,batch_size=batch_size)
    yhat_test = s.predict(X_test, apply_scale_to_X=False,batch_size=batch_size)
    result_dict={
                "train":{"y":y_scaler.inverse_transform(y_train), "yhat":yhat_train},
                "test":{"y":y_scaler.inverse_transform(y_test), "yhat":yhat_test}
                }
    columns = y.columns.tolist()
    plot_y_vs_yhat(columns,result_dict)

    return

# =============================================================================
# PHT3D twin (MODFLOW-2005 + MT3DMS + PHREEQC-2) — the benchmark counterpart to
# the mf6rtm reactive model, built from the same grid, flow field and chemistry.
# Gotchas that cost real debugging time are documented at each site below.
# =============================================================================

BIN_DIR = os.path.join('bin', 'mac' if platform.system() == 'Darwin' else 'win')
MF2005_EXE = os.path.join(BIN_DIR, 'mf2005')
PHT3D_EXE = os.path.join(BIN_DIR, 'pht3d')
MF6_REACTIVE_WS = os.path.join('model', 'reactive')   # mf6rtm reference run
PHT3D_WS = os.path.join('model', 'pht3d')
PHT3D_NAME = 'dizon_pht3d'    # flow twin
PHT3D_TR_NAME = 'pht3d_tr'    # transport; must differ from the flow model, else
                              # mt.write_input() overwrites the MODFLOW name file
POROSITY = 0.35

# FSP species-table schema (dependencies/pht3d_fsp/pht3d_species.xlsx)
_FSP_COLS = (["name", "initial_concentration", "species", "argument", "type", "mobility",
              "ion_exchange", "exchange_stoichiometry", "exchange_master_species",
              "surface_area", "surface_mass", "surface_phase", "surface_switch", "formula"]
             + [f"parameter_{i:02d}" for i in range(1, 101)])

# (ic_aq_chem var, fsp name, phreeqc species, mobility, argument)
_AQUEOUS = [
    ("O(0)", "o0", "O(0)", "mobile", None),
    ("N(+5)", "no3", "N(5)", "mobile", None),
    ("N(+3)", "n3", "N(3)", "mobile", None),
    ("N(0)", "n0", "N(0)", "mobile", None),
    ("S(6)", "so4", "S(6)", "mobile", None),
    ("S(-2)", "s2", "S(-2)", "mobile", None),
    ("C(+4)", "c4", "C(4)", "mobile", None),
    ("C(-4)", "ch4", "C(-4)", "mobile", None),
    ("Ca", "ca", "Ca", "mobile", None),
    ("Cl", "cl", "Cl", "mobile", None),
    ("Fe(+2)", "fe2", "Fe(2)", "mobile", None),
    ("Fe(+3)", "fe3", "Fe(3)", "mobile", None),
    ("K", "k", "K", "mobile", None),
    ("Mg", "mg", "Mg", "mobile", None),
    ("Na", "na", "Na", "mobile", None),
    ("Si", "si", "Si", "mobile", None),
    ("Amm", "amm", "Amm", "mobile", None),
    ("Tmp", "tmp", "Tmp", "mobile", None),
    # `charge` mirrors mup3d's `pH ... charge`. Tested against a plain `pH` line (what
    # PHT3D-FSP's own example writes): indistinguishable, max |diff| 3e-4 across every
    # variable and well, so this flag is not what drives the WP1 pH offset.
    ("pH", "ph", "pH", "immobile", "charge"),
    ("pe", "pe", "pe", "immobile", None),
]
# master species X; stoichiometry/master left blank because PHT3D reads the SAME database as
# mf6rtm (data/datab.dat), whose EXCHANGE_SPECIES already define CaX2/FeX2/KX/MgX2/NaX
_EXCHANGERS = [("Ca_ex", "CaX2", 2), ("Fe_ex", "FeX2", 2), ("K_ex", "KX", 1),
               ("Mg_ex", "MgX2", 2), ("Na_ex", "NaX", 1)]


def mf6_output_times(ws):
    """End-of-timestep times from TDIS, one per output step (validated == MF6 totim)."""
    sim = flopy.mf6.MFSimulation.load(sim_ws=ws, verbosity_level=0, load_only=["dis"])
    t, out = 0.0, []
    for perlen, nstp, tsmult in [(r[0], int(r[1]), r[2])
                                 for r in sim.tdis.perioddata.get_data()]:
        if tsmult == 1.0:
            dts = [perlen / nstp] * nstp
        else:
            dt0 = perlen * (tsmult - 1.0) / (tsmult ** nstp - 1.0)
            dts = [dt0 * tsmult ** i for i in range(nstp)]
        for dt in dts:
            t += dt
            out.append(t)
    return out


def pht3d_build_flow_twin(ws=PHT3D_WS, mf6_ws=MF6_REACTIVE_WS):
    """MODFLOW-2005 twin of the MF6 GWF, plus the LMT link file MT3DMS/PHT3D consume."""
    sim6 = flopy.mf6.MFSimulation.load(sim_ws=mf6_ws, verbosity_level=0)
    gwf = sim6.get_model("gwf")
    d = gwf.dis
    nlay, nrow, ncol = int(d.nlay.data), int(d.nrow.data), int(d.ncol.data)

    npers = int(sim6.tdis.nper.data)
    pdata = sim6.tdis.perioddata.array
    perlen = [float(r[0]) for r in pdata]
    nstp = [int(r[1]) for r in pdata]
    tsmult = [float(r[2]) for r in pdata]

    idomain = d.idomain.array
    ibound = np.where(idomain > 0, 1, 0).astype(int) if idomain is not None else 1
    npf = gwf.get_package("npf")
    hk = npf.k.array
    k33 = npf.k33.array if npf.k33 is not None and npf.k33.array is not None else hk
    icelltype = np.asarray(npf.icelltype.array)
    # MF2005 laytyp is per-LAYER; MF6 icelltype is per-cell. The model is confined
    # (icelltype all 0), so this collapses to laytyp = 0 everywhere.
    laytyp = np.array([1 if np.any(icelltype[k] != 0) else 0 for k in range(nlay)], dtype=int)
    strt = gwf.get_package("ic").strt.array
    sto = gwf.get_package("sto")
    ss = sto.ss.array if sto.ss.array is not None else 1e-5
    sy = sto.sy.array if sto.sy.array is not None else 0.15

    os.makedirs(ws, exist_ok=True)
    mf = flopy.modflow.Modflow(PHT3D_NAME, model_ws=ws, version="mf2005",
                               exe_name=os.path.abspath(MF2005_EXE))
    flopy.modflow.ModflowDis(
        mf, nlay=nlay, nrow=nrow, ncol=ncol,
        delr=d.delr.array, delc=d.delc.array, top=d.top.array, botm=d.botm.array,
        nper=npers, perlen=perlen, nstp=nstp, tsmult=tsmult, steady=[False] * npers,
        itmuni=4, lenuni=2,   # days, meters
    )
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    flopy.modflow.ModflowLpf(mf, hk=hk, vka=k33, laytyp=laytyp, ss=ss, sy=sy, ipakcb=53)

    chd6 = gwf.get_package("chd").stress_period_data.get_data()
    chd_spd = {}
    for per, recs in chd6.items():
        if recs is None:
            continue
        rows = []
        for rec in recs:
            lay, row, col = rec["cellid"]
            h = float(rec["head"])
            rows.append([lay, row, col, h, h])
        chd_spd[int(per)] = rows
    flopy.modflow.ModflowChd(mf, stress_period_data=chd_spd)

    welin6 = gwf.get_package("welin").stress_period_data.get_data()
    welout6 = gwf.get_package("welout").stress_period_data.get_data()
    wel_spd = {}
    for per in range(npers):
        rows = []
        for src in (welin6, welout6):
            recs = src.get(per)
            if recs is None:
                continue
            for rec in recs:
                lay, row, col = rec["cellid"]
                rows.append([lay, row, col, float(rec["q"])])
        if rows:
            wel_spd[per] = rows
    flopy.modflow.ModflowWel(mf, stress_period_data=wel_spd, ipakcb=53)

    flopy.modflow.ModflowPcg(mf, hclose=1e-7, rclose=1e-3, mxiter=200, iter1=100)
    flopy.modflow.ModflowOc(
        mf, stress_period_data={(p, nstp[p] - 1): ["save head", "save budget"]
                                for p in range(npers)})
    flopy.modflow.ModflowLmt(mf, output_file_name="mt3d_link.ftl")
    mf.write_input()
    return mf, gwf


def pht3d_compare_heads(ws=PHT3D_WS, mf6_ws=MF6_REACTIVE_WS, atol=1e-2):
    """Gate the flow twin against MF6 at matching simulation times.

    The twin saves once per stress period (39) while MF6 saves every timestep (854), so the
    files must be paired on `totim`. Pairing by index compares end-of-period heads against
    MF6's first 39 daily steps and reports a 0.7 m mismatch that does not exist.
    """
    f2005 = flopy.utils.HeadFile(os.path.join(ws, f"{PHT3D_NAME}.hds"))
    f6 = flopy.utils.HeadFile(os.path.join(mf6_ws, "gwf.hds"))
    t6 = np.asarray(f6.get_times())
    worst, tot, cnt = 0.0, 0.0, 0
    for t in f2005.get_times():
        j = int(np.argmin(np.abs(t6 - t)))
        if abs(t6[j] - t) > 1e-6:
            continue
        a = np.asarray(f2005.get_data(totim=t))
        b = np.asarray(f6.get_data(totim=t6[j]))
        mask = np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 1e29) & (np.abs(b) < 1e29)
        dif = np.abs(a[mask] - b[mask])
        worst = max(worst, float(dif.max()))
        tot += float(dif.sum())
        cnt += int(dif.size)
    print(f"flow twin vs MF6: max|dh|={worst:.4e} mean|dh|={tot / max(1, cnt):.4e} "
          f"(atol={atol}) -> {'PASS' if worst <= atol else 'CHECK'}")
    return worst <= atol


def pht3d_species_csv(ws=PHT3D_WS):
    """Author the FSP species table from dizon36's chemistry (single source of truth).

    ROW ORDER IS THE PHT3D COMPONENT ORDER (BTN sconc / SSM css / UCN numbering). PHT3D
    numbers components in the order pht3d_ph.dat blocks are read (multic.c::mcrp_):
    kinetic mobile (A) -> LEA aqueous (B, pH/pe last) -> kinetic immobile (C) ->
    equilibrium minerals (D) -> exchangers (E) -> surfaces (F) -> kinetic minerals (G).
    FSP writes ph.dat by type but keeps BTN arrays in row order, so the table must already
    be in PHT3D's order or transport and chemistry silently mismatch.
    """
    aq = pd.read_csv(os.path.join(datadir, "ic_aq_chem.csv")).set_index("var")["value"]
    surf = pd.read_csv(os.path.join(datadir, "ic_surfaces.csv"))
    surf1 = surf[surf["layer"] == 1].set_index("var")["value"]
    exch = pd.read_csv(os.path.join(datadir, "ic_exchanger.csv"))
    exch1 = exch[exch["layer"] == 1].set_index("var")["value"]

    rows = []

    def row(**kw):
        r = {c: np.nan for c in _FSP_COLS}
        r.update(kw)
        rows.append(r)

    # comp 1: Orgc, kinetic mobile (type A). It is a SOLUTION_MASTER_SPECIES in datab.dat
    # (so it is transported) AND the kinetic-rate reactant. `argument` carries m0: the rate
    # scales with (m/m0) and PHT3D falls back to m0 = 10 when absent, so it must be the
    # mf6rtm value (kin_dic m0 = 1.0).
    row(name="orgc", initial_concentration=float(aq.get("Orgc", 0.0)), species="Orgc", type="A",
        mobility="mobile", ion_exchange="no", formula="Orgc -1.0 CH2O 1.0", argument=1.0,
        parameter_01=1.57e-9, parameter_02=1.67e-11, parameter_03=1.0e-13)

    # comps 2..19: mobile aqueous (type B); comps 20-21: pH, pe (last two of the aqueous block)
    for want_mobile in (True, False):
        for var, name, species, mob, arg in _AQUEOUS:
            if (mob == "mobile") != want_mobile:
                continue
            row(name=name, initial_concentration=float(aq.get(var, 0.0)), species=species,
                argument=arg, type="B", mobility=mob, ion_exchange="no")

    # equilibrium minerals (SI in `argument`, m0 in initial_concentration). MUST match the
    # `eq_keys` list in initialize_chemistry -- these two are the same physical choice expressed
    # in each code's dialect, and a mismatch makes the codes incomparable.
    for var in ("Ferrihydrite", "Orgmatter"):
        row(name=var.lower(), initial_concentration=float(surf1.get(var, 0.0)), species=var,
            argument=0.0, type="D", mobility="immobile", ion_exchange="no")

    # comps 23..27: ion exchangers (type E)
    for var, sp, _stoich in _EXCHANGERS:
        row(name=sp.lower(), initial_concentration=float(exch1.get(var, 0.0)), species=sp,
            type="E", mobility="immobile", ion_exchange="yes")

    # comp 28 (last): kinetic mineral Pyrite (type G)
    row(name="pyrite", initial_concentration=float(surf1.get("Pyrite", 0.0)), species="Pyrite",
        type="G", mobility="immobile", ion_exchange="no",
        parameter_01=16.0, parameter_02=0.67, parameter_03=0.5, parameter_04=-0.11)

    df = pd.DataFrame(rows, columns=_FSP_COLS)
    out = os.path.join(ws, "pht3d_species.csv")
    os.makedirs(ws, exist_ok=True)
    df.to_csv(out, index=False)
    return df, out


def pht3d_equilibrated_exchangers(sout=os.path.join(MF6_REACTIVE_WS, "sout.csv")):
    """Per-layer exchanger composition after PHREEQC's initial equilibration, from mf6rtm.

    mf6rtm's EXCHANGE blocks carry `-equilibrate 1`, so PHREEQC redistributes the cations
    against the background solution before transport starts. PHT3D has no equivalent step and
    takes the listed amounts as given, which leaves its exchanger under-loaded with Fe (the raw
    ic_exchanger.csv FeX2 is 1.567x below the equilibrated value in layers 7-12). The exchanger
    then strips Fe(2) from solution and displaces Ca/Mg into it.

    Returns {sout column: {1-based layer: mol/L water}}, or None if sout.csv is missing. The
    first output time is one day in -- far closer to the equilibrated state than the raw values;
    the per-layer median keeps well-adjacent cells from skewing it.
    """
    if not os.path.exists(sout):
        return None
    cols = ["MOL_CaX2", "MOL_FeX2", "MOL_KX", "MOL_MgX2", "MOL_NaX"]
    # the first output time is the leading nlay*ncell block, so a bounded read suffices on a
    # multi-GB sout.csv
    s = pd.read_csv(sout, usecols=["time", "layer"] + cols, nrows=60000)
    s = s[s["time"] == s["time"].min()]
    med = s.groupby("layer")[cols].median()
    return {c: med[c].to_dict() for c in cols}


def _pht3d_rewrite_ssm(path, ssm, npers, ncomp, itype=2):
    """Rewrite the SSM point-source records in the hybrid format PHT3D actually reads.

    mt_ssm5.for reads each record as (3I10,F10.0,I10) with ADVANCE='NO' and then takes the
    per-component concentrations list-directed from the rest of the line. flopy instead writes
    every field fixed-width %10G with no delimiter, so a value needing 11 characters (any of the
    7-digit background concentrations) runs into its neighbour: the integer read hits a malformed
    token and PHT3D dies with a Fortran error termination.

    `css` (the fixed F10.0 field) is written as 0 -- PHT3D only consults it when negative, which
    flags a recirculation well; the real concentrations are the trailing list.
    """
    with open(path) as f:
        head = f.readlines()[:2]   # source-type flags + MXSS
    with open(path, "w") as f:
        f.writelines(head)
        for per in range(npers):
            rows = ssm.get(per, [])
            f.write(f"{len(rows):10d}{0:10d} # stress period {per + 1}\n")
            for r in rows:
                k, i, j = int(r[0]), int(r[1]), int(r[2])
                css_all = r[5:]
                assert len(css_all) == ncomp, f"{len(css_all)} css values, expected {ncomp}"
                fixed = f"{k + 1:10d}{i + 1:10d}{j + 1:10d}{0.0:10.1f}{itype:10d}"
                f.write(fixed + " " + " ".join(f"{v:.6e}" for v in css_all) + "\n")


def pht3d_build_transport(ws=PHT3D_WS):
    """MT3DMS/PHT3D transport deck + PHREEQC reaction file on the flow twin's FTL."""
    import sys
    sys.path.insert(0, os.path.join("dependencies", "pht3d_fsp"))
    import pht3d_fsp

    mf = flopy.modflow.Modflow.load(f"{PHT3D_NAME}.nam", model_ws=ws, version="mf2005",
                                    exe_name=os.path.abspath(MF2005_EXE), verbose=False,
                                    check=False)
    nlay, nrow, ncol = mf.nlay, mf.nrow, mf.ncol
    npers = mf.dis.nper
    perlen = np.asarray(mf.dis.perlen.array, dtype=float)

    spec = pht3d_fsp.create(xlsx_path=ws + "/", xlsx_name="pht3d_species.csv",
                            nlay=nlay, nrow=nrow, ncol=ncol, pht3d_path=ws + "/")
    order = list(spec.keys())
    ncomp = int(pht3d_fsp.create.ncomp)
    mcomp = int(pht3d_fsp.create.mcomp)

    surf = pd.read_csv(os.path.join(datadir, "ic_surfaces.csv"))
    exch = pd.read_csv(os.path.join(datadir, "ic_exchanger.csv"))
    exmap = {"cax2": "Ca_ex", "fex2": "Fe_ex", "kx": "K_ex", "mgx2": "Mg_ex", "nax": "Na_ex"}
    soutname = {"cax2": "CaX2", "fex2": "FeX2", "kx": "KX", "mgx2": "MgX2", "nax": "NaX"}
    surfmap = {"ferrihydrite": "Ferrihydrite", "orgmatter": "Orgmatter", "pyrite": "Pyrite"}
    equil = pht3d_equilibrated_exchangers()

    # PHT3D divides immobile-species BTN values by porosity itself (verified: feeding Pyrite
    # 0.3753 yields 1.0722 = 0.3753/0.35 in PHT3D028.UCN), whereas PHREEQC uses the numbers in
    # mf6rtm's phinp.dat as-is. To land on the same per-litre-of-water amounts as mf6rtm:
    #   minerals   -- mf6rtm converts vol-bulk -> vol-water, so feed the RAW vol-bulk value and
    #                 let PHT3D do that conversion;
    #   exchangers -- mf6rtm writes the raw value straight into EXCHANGE, so pre-multiply by
    #                 porosity to cancel PHT3D's division.
    # Getting this wrong left Pyrite 1/0.35 = 2.86x too abundant; its rate law contains
    # log10(m*115), so nitrate-fed pyrite oxidation ran far too fast and NO3 came out ~10x low.
    sconc = {}
    for i, nm in enumerate(order):
        arr = np.array(spec[nm], dtype=float)
        if nm in surfmap:
            col = surf[surf["var"] == surfmap[nm]].set_index("layer")["value"]
            for lay in range(nlay):
                arr[lay, :, :] = float(col.get(lay + 1, 0.0))
        elif nm in exmap:
            col = exch[exch["var"] == exmap[nm]].set_index("layer")["value"]
            eqcol = (equil or {}).get(f"MOL_{soutname[nm]}")
            for lay in range(nlay):
                v = float(col.get(lay + 1, 0.0))
                if eqcol is not None:   # prefer mf6rtm's post-equilibration composition
                    v = float(eqcol.get(lay + 1, v))
                arr[lay, :, :] = v * POROSITY
        sconc[i + 1] = arr

    mt = flopy.mt3d.Mt3dms(modelname=PHT3D_TR_NAME, model_ws=ws, version="mt3dms",
                           exe_name=os.path.abspath(PHT3D_EXE), modflowmodel=mf,
                           ftlfilename="mt3d_link.ftl")
    timprs = np.cumsum(perlen)
    sconc_kw = {"sconc": sconc[1]}
    for ic in range(2, ncomp + 1):
        sconc_kw[f"sconc{ic}"] = sconc[ic]
    # icbund follows ibound: with icbund=1 in flow-inactive cells PHREEQC is handed cells
    # holding no mass, and charge-balancing pH there diverges until the run aborts.
    icbund = np.where(np.asarray(mf.bas6.ibound.array) > 0, 1, 0)
    flopy.mt3d.Mt3dBtn(mt, ncomp=ncomp, mcomp=mcomp, prsity=POROSITY, icbund=icbund,
                       species_names=order, nprs=len(timprs), timprs=timprs,
                       tunit="D", lunit="M", munit="mol", **sconc_kw)
    flopy.mt3d.Mt3dAdv(mt, mixelm=-1, percel=1.0)   # TVD
    flopy.mt3d.Mt3dDsp(mt, al=0.1, trpt=0.1, trpv=0.01, dmcoef=0.0)
    flopy.mt3d.Mt3dGcg(mt, mxiter=50, iter1=50, isolve=3, cclose=1e-6)

    # Heat retardation. mf6rtm gives Tmp linear sorption (bulk_density 1850,
    # distcoef 2.1141e-4 -> R = 1 + rhob*Kd/theta ~ 2.12), and both the Orgc and Pyrite rate
    # laws in data/datab.dat scale by an Arrhenius factor from tot("Tmp"), so without this the
    # twin's redox chain runs on a thermal front arriving twice too early. Every component
    # except Tmp keeps Kd = 0. (Worth having for correctness; it moved NO3 by under 2 %.)
    tmp_comp = order.index("tmp") + 1
    flopy.mt3d.Mt3dRct(mt, isothm=1, ireact=0, igetsc=0, rhob=1850.0, sp1=0.0,
                       **{f"sp1{tmp_comp}": 2.1141e-4})

    # --- SSM: per-period well injection from wellin.csv, plus the extraction wells ---
    win = pd.read_csv(os.path.join(datadir, "wellin.csv"))
    colmap = {name: var for (var, name, sp, mob, arg) in _AQUEOUS}
    colmap["orgc"] = "Orgc"
    welspd = mf.wel.stress_period_data.data
    # Extraction wells need a chemically valid source solution even though MT3DMS ignores css
    # at a sink: PHT3D instantiates one PHREEQC solution per SSM entry, and an all-zero entry is
    # pure water, where charge-balancing pH diverges (pH 0, 100 % charge error) and aborts the
    # run mid-simulation. mf6rtm gives welout the background solution, so do the same. Minerals
    # and exchangers stay 0 -- they are not part of a well's solution.
    spec_tbl = pd.read_csv(os.path.join(ws, "pht3d_species.csv")).set_index("name")
    background = [0.0 if spec_tbl.loc[nm, "type"] in ("D", "E", "G")
                  else float(spec_tbl.loc[nm, "initial_concentration"]) for nm in order]
    ssm = {}
    for per in range(npers):
        recs = welspd.get(per)
        if recs is None:
            continue
        rows = []
        wcsv = win[win["kper"] == per]
        for rec in recs:
            k, i, j, q = int(rec["k"]), int(rec["i"]), int(rec["j"]), float(rec["flux"])
            css_all = list(background)
            if q > 0:
                # make_wel_in uses the layer values (1,2,3,5,7) directly as 0-based cell layer
                # indices, so the well cell's 0-based k matches wellin.csv's 'layer' as-is.
                lrow = wcsv[wcsv["layer"] == k]
                if len(lrow):
                    for idx, nm in enumerate(order):
                        col = colmap.get(nm)
                        if col is not None and col in lrow.columns:
                            css_all[idx] = float(lrow[col].values[0])
            rows.append([k, i, j, css_all[0], 2] + css_all)
        ssm[per] = rows
    flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm)

    mt.write_input()
    _pht3d_rewrite_ssm(os.path.join(ws, f"{PHT3D_TR_NAME}.ssm"), ssm, npers, ncomp)

    # PHT3D only invokes PHREEQC if the name file has a PHC entry pointing at pht3d_ph.dat.
    # flopy's Mt3dms writer knows nothing about the package, so without this the binary runs as
    # plain MT3DMS and silently skips ALL chemistry.
    nam = os.path.join(ws, f"{PHT3D_TR_NAME}.nam")
    with open(nam) as f:
        txt = f.read().rstrip("\n")
    if "PHC" not in txt:
        with open(nam, "w") as f:
            f.write(txt + "\nPHC               64  pht3d_ph.dat\n")
    return mt, ncomp, mcomp


def pht3d_run(ws=PHT3D_WS, clean=True):
    """Run the PHT3D binary (~9 min for 39 periods / 854 d)."""
    if clean:
        pats = ("PHT3D0", ".MAS", ".XMAS", "fort.", "MT3D.CNF", "phinp.dat", "phout.dat",
                "phreeqc.log", "pht3d_input_check.dat")
        for f in os.listdir(ws):
            if any(p in f for p in pats):
                os.remove(os.path.join(ws, f))
    pyemu.os_utils.run(f"{os.path.relpath(os.path.abspath(PHT3D_EXE), ws)} "
                       f"{PHT3D_TR_NAME}.nam", cwd=ws)


def pht3d_extract(ws=PHT3D_WS, mf6_ws=MF6_REACTIVE_WS):
    """PHT3D UCNs -> data/pht3dout.csv (time,variable,wp,value), the digitized-file schema.

    Component order is the row order of pht3d_species.csv, so PHT3D0NN.UCN is component NN.
    The mapping is cross-checked against pht3d_input_check.dat, which the binary writes while
    parsing pht3d_ph.dat and is therefore authoritative -- a re-ordered table cannot silently
    mislabel the output. Comparing UCN values against initial concentrations does NOT work: the
    first output time is already past a reaction step and O(0) starts at 1e-18.

    Note the unit difference: PHT3D UCNs are mol/L, MF6 GWT UCNs are mol/m3 (sout.csv mol/L).
    """
    spec = pd.read_csv(os.path.join(ws, "pht3d_species.csv"))
    idx = {nm: i + 1 for i, nm in enumerate(spec["name"])}
    spec = spec.set_index("name")

    reported = {}
    with open(os.path.join(ws, "pht3d_input_check.dat")) as f:
        for line in f:
            if line.startswith("Component "):
                num, rest = line[len("Component "):].split(":", 1)
                reported[int(num)] = rest.strip().rstrip(".").split(".")[0].strip()
    for name in PHT3D_VARS.values():
        comp, want, got = idx[name], str(spec.loc[name, "species"]), reported.get(idx[name])
        if got != want:
            raise AssertionError(
                f"component {comp} is '{got}' per pht3d_input_check.dat but the species table "
                f"says '{want}' ({name}) -- table order and PHT3D numbering disagree")

    cells = obs_well_cells(mf6_ws)
    rows = []
    for var, name in PHT3D_VARS.items():
        u = flopy.utils.UcnFile(os.path.join(ws, f"PHT3D{idx[name]:03d}.UCN"))
        for t in u.get_times():
            d = u.get_data(totim=t)
            for wp, c in cells.items():
                rows.append(dict(time=float(t), variable=var, wp=wp,
                                 value=float(d[c["lay"], c["row"], c["col"]])))
    df = pd.DataFrame(rows)
    out = os.path.join(datadir, "pht3dout.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df['time'].nunique()} times")
    return df


# =============================================================================
# Comparison figures: mf6rtm vs observations vs the PHT3D twin
# =============================================================================

VAR_LABEL = {'o0': 'DO', 'no3': 'NO3', 'so4': 'SO4', 'tic': 'TIC', 'ph': 'pH'}
VAR_UNIT = {'o0': '(mol L$^{-1}$)', 'no3': '(mol L$^{-1}$)', 'so4': '(mol L$^{-1}$)',
            'tic': '(mol L$^{-1}$)', 'ph': ''}
VAR_LIMITS = {'ph': [6, 8], 'so4': [0, 1.5e-3], 'no3': [0, 5.2e-4],
              'o0': [0, 1e-3], 'tic': [0, 0.01]}
# data/pht3dout.csv (and the digitized file) variable -> plot variable
PHT3D_MAP = {'DO': 'o0', 'N5': 'no3', 'S6': 'so4', 'C4': 'tic', 'pH': 'ph'}
# plot variable -> species-table row name, for pht3d_extract
PHT3D_VARS = {'DO': 'o0', 'N5': 'no3', 'S6': 'so4', 'C4': 'c4', 'pH': 'ph'}

FIG_VARS = ['o0', 'no3', 'so4', 'tic', 'ph']
FIG_WPS = ['wp3', 'wp2', 'wp1']
FIG_SCREEN = 'f2'


def obs_well_cells(mf6_ws=MF6_REACTIVE_WS, wells=('WP1', 'WP2', 'WP3'), screen='f2'):
    """{WPn: {lay,row,col,flat1}} for each well screen, from data/obs_loc.csv."""
    sim = flopy.mf6.MFSimulation.load(sim_ws=mf6_ws, verbosity_level=0)
    gwf = sim.get_model("gwf")
    nrow, ncol = int(gwf.dis.nrow.data), int(gwf.dis.ncol.data)
    ix = GridIntersect(gwf.modelgrid)
    obsloc = pd.read_csv(os.path.join(datadir, "obs_loc.csv"))
    cells = {}
    for wp in wells:
        r = obsloc[obsloc.obsid == f"{wp}-{screen}"].iloc[0]
        row, col = ix.intersect([(r.x, r.y)], shapetype="point").cellids[0]
        lay = int(r.layer)
        cells[wp] = dict(lay=lay, row=int(row), col=int(col),
                         flat1=lay * (nrow * ncol) + int(row) * ncol + int(col) + 1)
    return cells


def load_mf6rtm_series(ws=MF6_REACTIVE_WS):
    """sout.csv rows at the observation cells, tagged with well and screen.

    Handles both sout.csv schemas. mf6rtm before commit 7828ece wrote `time,cell,<vars>` with a
    0-based `cell` and stamped time as (ctime + dt) -- one timestep AHEAD of the row's state.
    Newer versions write `time,cell,layer,row,col,<vars>` with a 1-based `cell` and the correct
    time. Reading a legacy file without both corrections samples one cell off AND shifts every
    curve one step right (28-35 d late in this model). See docs/mf6rtm_time_axis_bug.md.
    """
    sim = flopy.mf6.MFSimulation.load(sim_ws=ws, verbosity_level=0)
    gwf = sim.get_model("gwf")
    nrow, ncol = int(gwf.dis.nrow.data), int(gwf.dis.ncol.data)
    ix = GridIntersect(gwf.modelgrid)
    obsloc = pd.read_csv(os.path.join(datadir, "obs_loc.csv"))
    flat = []
    for obsid in obsloc.obsid.unique():
        x, y = obsloc.loc[obsloc.obsid == obsid, ['x', 'y']].values[0]
        cid = ix.intersect([(x, y)], shapetype="point").cellids
        if len(cid) == 0:
            continue
        r, c = cid[0]
        lay = int(obsloc.loc[obsloc.obsid == obsid, 'layer'].values[0])
        flat.append((obsid.lower(), lay * (nrow * ncol) + r * ncol + c))

    sout = pd.read_csv(os.path.join(ws, "sout.csv"))
    sout.columns = [c.strip().lower() for c in sout.columns]
    legacy = not all(k in sout.columns for k in ('layer', 'row', 'col'))
    lookup = {(fi if legacy else fi + 1): oid for oid, fi in flat}
    sout['cell'] = sout['cell'].astype(float).astype(int)
    if legacy:
        true_t = mf6_output_times(ws)
        seen = list(dict.fromkeys(sout['time']))
        if len(seen) == len(true_t):
            sout['time'] = sout['time'].map(dict(zip(seen, true_t)))
        else:
            print(f"WARNING {ws}: {len(seen)} output steps vs {len(true_t)} TDIS steps; "
                  "time axis left uncorrected")
    sout['obsid'] = sout['cell'].map(lookup)
    sel = sout.dropna(subset=['obsid']).copy()
    sel['wp'] = sel['obsid'].str.extract(r'((?:wp|ip|pp)\d+)', expand=False)
    sel['f'] = sel['obsid'].str.extract(r'(f\d+)', expand=False)
    return sel


def load_observations():
    m = pd.read_csv(os.path.join(datadir, "obs_chem_cleaned.csv"))
    m['variable'] = m['variable'].str.lower()
    m['wp'] = m['obsid'].str.extract(r'((?:wp|ip|pp)\d+)', expand=False)
    m['f'] = m['obsid'].str.extract(r'(f\d+)', expand=False)
    return m


def load_pht3d_out():
    p = pd.read_csv(os.path.join(datadir, "pht3dout.csv"))
    p['pvar'] = p['variable'].map(PHT3D_MAP)
    p['wp'] = p['wp'].str.lower()
    return p


def plot_comparison(show_obs=True, show_pht3d=True, ws=MF6_REACTIVE_WS, out=None):
    """5x3 panel (rows DO/NO3/SO4/TIC/pH, cols WP3/WP2/WP1) at screen f2.

    mf6rtm is a solid line and observations are filled dots. PHT3D is drawn as open circles when
    it is the only thing compared against mf6rtm, but as a dashed line on the three-way figure,
    where circles compete with the observation markers -- there mf6rtm is also thickened so the
    two model curves stay separable.
    """
    import matplotlib.ticker as mticker
    sel = load_mf6rtm_series(ws)
    meas = load_observations() if show_obs else None
    pht = load_pht3d_out() if show_pht3d else None
    if out is None:
        tag = {(True, True): 'obs_pht3d', (True, False): 'obs', (False, True): 'pht3d'}[
            (show_obs, show_pht3d)]
        out = os.path.join("output", f"dizon_mf6rtm_vs_{tag}.png")
    three_way = show_obs and show_pht3d

    fig, axes = plt.subplots(len(FIG_VARS), len(FIG_WPS), figsize=(8, 8.5),
                             sharex='col', sharey='row')
    axes = np.atleast_2d(axes)
    for r, var in enumerate(FIG_VARS):
        for c, wp in enumerate(FIG_WPS):
            ax = axes[r, c]
            line = sel[(sel['wp'] == wp) & (sel['f'] == FIG_SCREEN)].sort_values('time')
            ax.plot(line['time'], line[var], color='tab:blue',
                    lw=2.2 if three_way else 1.3, label='mf6rtm')
            if pht is not None:
                pp = pht[(pht['wp'] == wp) & (pht['pvar'] == var)].sort_values('time')
                if three_way:
                    ax.plot(pp['time'], pp['value'], ls='--', color='tab:red', lw=1.3,
                            label='PHT3D')
                else:
                    ax.plot(pp['time'], pp['value'], 'o', mfc='none', mec='tab:red', mew=1.1,
                            ms=4.5, label='PHT3D')
            if meas is not None:
                mp = meas[(meas['wp'] == wp) & (meas['f'] == FIG_SCREEN)
                          & (meas['variable'] == var)]
                if three_way:   # filled, to stay distinct from the dashed PHT3D line
                    ax.scatter(mp['time'], mp['value'], c='k', s=18, zorder=10, label='observed')
                else:
                    ax.scatter(mp['time'], mp['value'], facecolors='none', edgecolors='k',
                               linewidths=1.1, s=22, zorder=10, label='observed')
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 2))
            if r == 0:
                ax.set_title(wp.upper())
            if c == 0:
                ax.set_ylabel(f"{VAR_LABEL[var]} {VAR_UNIT[var]}")
            if r == len(FIG_VARS) - 1:
                ax.set_xlabel('Time (days)')
            ax.set_ylim(VAR_LIMITS[var])
            ax.set_xlim(0, 875)
    axes[0, -1].legend(fontsize=7, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out}")
    return out


def make_figures(ws=MF6_REACTIVE_WS):
    """The three comparison figures: vs data, vs PHT3D, and vs both."""
    return [plot_comparison(show_obs=True, show_pht3d=False, ws=ws),
            plot_comparison(show_obs=False, show_pht3d=True, ws=ws),
            plot_comparison(show_obs=True, show_pht3d=True, ws=ws)]


def main(prep_obs = True, run_base = True, run_base_struct=False,
         build_pht3d=False, run_pht3d=False, extract_pht3d=False, figures=False,
         prep_pest = False, run_pest = False):

    if prep_obs:
        clean_obs_chem(datadir = "data",
                        input_path="obs_chem_raw_0.csv", 
                    output_path="obs_chem_cleaned.csv")
    
    if run_base:
        ws = prep_model_dir(name='reactive_demo')
        nlay = 12
        nrow = 10
        ncol = 51
        mup3d_m=initialize_chemistry(ws, nlay, nrow, ncol)
        tracer = None

        sim = make_gwf(ws, tracer=tracer,mup3d_m=mup3d_m)
        sim = make_gwt(sim, tracer=tracer, mup3d_m=mup3d_m)

        pyemu.os_utils.run('mf6rtm', cwd=sim.sim_path)
    if run_base_struct:
        ws = prep_model_dir(name='reactive')
        # --- flopy MF6: GWF (Cl-aux wells + CHD) + ONE conservative Cl tracer GWT ---
        sim = make_gwf_structured(ws, tracer='Cl', mup3d_m=None, write=False)
        sim = make_gwt(sim, tracer='Cl', mup3d_m=None, write=False)   # single Cl tracer GWT = from_mf6 template (in-memory, no mid-build write)
        gwf = sim.get_model("gwf")
        nlay = gwf.dis.nlay.get_data()
        nrow = gwf.dis.nrow.get_data()
        ncol = gwf.dis.ncol.get_data()
        # dizon36's make_wel_*/make_chd store SPD as externalized pandas-backed lists whose
        # column count is locked. mf6rtm.from_mf6 grows the well/CHD auxiliary from 1 (the Cl
        # tracer) to ncomp (one column per PHREEQC component); the locked form rejects that
        # ("expected 3 got 18"). Rebuild these 3 packages as fresh in-memory (resizable) lists
        # so the aux expansion in write_simulation() succeeds. See docs/from_mf6_notes.md.
        _stress = {'welin': flopy.mf6.ModflowGwfwel,
                   'welout': flopy.mf6.ModflowGwfwel,
                   'CHD': flopy.mf6.ModflowGwfchd}
        for _pn, _cls in _stress.items():
            _p = gwf.get_package(_pn)
            _spd = _p.stress_period_data.get_data()
            gwf.remove_package(_pn)
            _cls(gwf, stress_period_data=_spd, auxiliary='Cl', pname=_pn, save_flows=True)
        # --- attach PHREEQC chemistry via from_mf6 (clones the Cl GWT into one reactive GWT per component) ---
        mup3d_m = initialize_chemistry(ws, nlay, nrow, ncol, sim=sim, gwt_name='tracer')
        # --- boundary injection chemistry (reproduces make_wel_in / make_chd mapping) ---
        # welin: 5 layers x 39 periods -> solutions 2..196 (5 per period); welout: extraction; chd: background solution 1
        inj_idx = [list(range(i, i + 5)) for i in range(2, 197, 5)]
        def _ncells(pkgname):
            spd = gwf.get_package(pkgname).stress_period_data.get_data()
            return len(spd[sorted(spd)[0]])
        cs_welin = mup3d.ChemStress('welin', type='aux')
        cs_welin.set_spd({per: inj_idx[per] for per in range(nper)})
        mup3d_m.set_chem_stress(cs_welin)
        cs_welout = mup3d.ChemStress('welout', type='aux')
        cs_welout.set_spd([1] * _ncells('welout'))   # extraction: injected conc ignored by MF6
        mup3d_m.set_chem_stress(cs_welout)
        cs_chd = mup3d.ChemStress('CHD', type='aux')
        cs_chd.set_spd([1] * _ncells('CHD'))          # background solution 1 (pname is 'CHD')
        mup3d_m.set_chem_stress(cs_chd)
        # --- per-component MST: Tmp is heat, and heat retards (kd = 2*ne/1850) ---
        # from_mf6 clones ONE tracer MST into every component, so without this override the
        # Tmp GWT loses its linear sorption and heat travels unretarded. Pyrite kinetics are
        # temperature-dependent, so a wrong temperature field propagates into the whole
        # redox chain. Values match make_gwt's `if comp=='Tmp'` branch.
        mup3d_m.set_mst_override({'Tmp': {'sorption': 'Linear',
                                          'bulk_density': 1850.0,
                                          'distcoef': 2.1141E-04}})
        # --- write coupled sim + run mf6rtm ---
        mup3d_m.write_simulation()
        pyemu.os_utils.run('mf6rtm', cwd=mup3d_m.wd)
    if build_pht3d:
        # PHT3D twin of the mf6rtm run in model/reactive. The flow twin is gated against MF6
        # heads before the transport deck is built on its FTL.
        pht3d_build_flow_twin()
        mf = flopy.modflow.Modflow.load(f"{PHT3D_NAME}.nam", model_ws=PHT3D_WS,
                                        version="mf2005",
                                        exe_name=os.path.abspath(MF2005_EXE), check=False)
        mf.run_model(silent=True)
        pht3d_compare_heads()
        pht3d_species_csv()
        pht3d_build_transport()
    if run_pht3d:
        pht3d_run()
    if extract_pht3d:
        pht3d_extract()
    if figures:
        make_figures()
    if prep_pest:
        template_ws=os.path.join('pest','pst_template')
        org_d = os.path.join('model','test')
        setup_pest(org_d, num_reals=100)
        set_obsval_and_weights()
        add_std_to_pst(fraction=0.05)
        build_noise_ensemble()
    if run_pest:
        md=os.path.join('pest','master1')
        run_pestpp(md=md, td=template_ws, casename="dizon36", 
                noptmax=1,freeze=True,
                num_reals=50,
                num_workers=15, worker_root=".", 
                pestpp_version="ies",restart=False,
                reuse_master=False, cleanup=True)

if __name__ == "__main__":
    main(
        prep_obs = False,
        run_base = False,
        run_base_struct = False,
        build_pht3d = False,
        run_pht3d = False,
        extract_pht3d = False,
        figures = False,
        prep_pest = True,
        run_pest = True
    )
    surrogate_workflow()
