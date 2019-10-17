import os, sys
import numpy as np
import pandas as pd
from ast import literal_eval
import re
import h0rton.tdlmc_data
__all__ = ['convert_to_dataframe', 'parse_closed_box', 'parse_open_box']

tdlmc_data_path = os.path.abspath(h0rton.tdlmc_data.__path__[0])
"""str: directory path containing the TDLMC data

"""

def convert_to_dataframe(rung, save_csv_path):
    """Store the TDLMC closed and open boxes into a Pandas DataFrame and exports to a csv file at the same location

    Parameters
    ----------
    rung : int
        rung number
    save_csv_path : str
        path of the csv file to be generated

    Returns
    -------
    Pandas DataFrame
        the extracted rung data

    """
    if save_csv_path is None:
        save_csv_path = os.path.join(tdlmc_data_path, 'rung{:d}_combined.csv'.format(rung))
        print("Saving rung {:d} data at {:s}...".format(rung, save_csv_path))

    df = pd.DataFrame()
    for code in ['code1', 'code2']:
        closed_code_dir = os.path.join(tdlmc_data_path, 'rung{:d}'.format(rung), code)
        open_code_dir = os.path.join(tdlmc_data_path, 'rung{:d}_open_box'.format(rung), code)
        seeds = sorted(os.listdir(closed_code_dir)) # list of seeds, e.g. 'f160w-seed101'
        row = {} # initialized dict in which to save lens info
        for seed in seeds:
            # Path to the text files
            closed_box_path = os.path.join(closed_code_dir, seed, 'lens_info_for_Good_team.txt')
            open_box_path = os.path.join(open_code_dir, seed, 'lens_all_info.txt')
            # Save seed path for easy access
            row['name'] = 'rung{:d}_{:s}_{:s}'.format(rung, code, seed)
            row['seed_path'] = os.path.join(closed_box_path, seed)
            # Parse the text files
            row = parse_closed_box(closed_box_path, row)
            row = parse_open_box(open_box_path, row)
            df = df.append(row, ignore_index=True)
    # Unravel nested dictionaries in some columns
    lens_mass = df['lens_mass'].apply(pd.Series).copy().add_prefix('lens_mass_')
    lens_light = df['lens_light'].apply(pd.Series).copy().add_prefix('lens_light_')
    ext_shear_bphi = df['ext_shear_bphi'].apply(pd.Series).copy().add_prefix('ext_shear_')
    ext_shear_e1e2 =  df['ext_shear_e1e2'].apply(pd.Series).copy().add_prefix('ext_shear_')
    df = pd.concat([df.drop(['lens_mass', 'lens_light', 'ext_shear_bphi', 'ext_shear_e1e2'], axis=1), lens_mass, lens_light, ext_shear_bphi, ext_shear_e1e2], axis=1)
    df.to_csv(save_csv_path, index=None)
    return df

def parse_closed_box(closed_box_path, row_dict=dict()):
    """Parse the lines of an open-box TDLMX text file for Rungs 0, 1, and 2

    Parameters
    ----------
    closed_box_path : str
        path to the closed box text file, `lens_info_for_Good_team.txt.txt`
    row_dict : dict
        dictionary of the row info to update. Default: dict()

    Returns
    -------
    dict
        An updated dictionary containing the information in the closed box text file

    """
    file = open(closed_box_path)
    lines = [line.rstrip('\n') for line in file]

    row_dict['z_lens'], row_dict['z_src'] = literal_eval(lines[2].split('\t')[1])
    row_dict['measured_vel_disp'] = float(lines[5].split('\t')[1].split('km/s')[0])
    row_dict['measured_vel_disp_err'] = float(lines[5].split('\t')[1].split('km/s')[1].split(':')[1])
    row_dict['measured_time_delays'] = literal_eval(re.split('\(|\)', lines[7])[1])
    row_dict['measured_time_delays_err'] = literal_eval(re.split('\(|\)', lines[7])[3])
    return row_dict

def parse_open_box(open_box_path, row_dict=dict()):
    """Parse the lines of an open-box TDLMX text file for Rungs 0, 1, and 2

    Parameters
    ----------
    open_box_path : str
        path to the open box text file, `lens_all_info.txt`
    row_dict : dict
        dictionary of the row info to update. Default: dict()

    Returns
    -------
    dict
        An updated dictionary containing the information in the open box text file

    """
    file = open(open_box_path)
    lines = [line.rstrip('\n') for line in file]
    row_dict['H0'] = float(re.split(':\s|km/s/Mpc', lines[3])[-2])
    row_dict['td_distance'] = float(re.split('ls:|Mpc', lines[5])[-2])
    row_dict['time_delays'] = literal_eval(re.split('\(|\)', lines[7])[1])
    row_dict['lens_mass'] = literal_eval(lines[11][7:])
    row_dict['ext_shear_e1e2'], row_dict['ext_shear_bphi'] = literal_eval(re.split('\(|\)', lines[12])[1])
    row_dict['lens_light'] = literal_eval(lines[14].split('\t')[1])
    row_dict['host_name'] = re.split('\(|\)|:|\t', lines[16])[2][1:]
    row_dict['host_pos'] = literal_eval(re.split('\(|\)|:|\t', lines[16])[-2])
    row_dict['host_mag'] = float(re.split('\t|\s', lines[17])[3])
    row_dict['host_r_eff'] = float(re.split('\t|\s', lines[17])[7])
    row_dict['agn_src_amp'] = float(lines[20].split()[-1])
    row_dict['agn_img_pos_x'] = literal_eval(re.split('\(|\)', lines[21])[1])
    row_dict['agn_img_pos_y'] = literal_eval(re.split('\(|\)', lines[21])[3])
    row_dict['agn_img_amp'] = literal_eval(re.split('\(|\)', lines[22])[1])
    row_dict['host_img_mag'] = re.split('plane: |mag|', lines[23])[3]
    row_dict['agn_img_mag'] = re.split('plane: |mag|', lines[23])[7]
    row_dict['vel_disp'] = float(re.split('km\/s| |\t', lines[25])[1])
    row_dict['kappa_ext'] = float(lines[27].split('\t')[1])

    return row_dict