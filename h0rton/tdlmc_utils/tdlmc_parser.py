import os
import numpy as np
import pandas as pd
from ast import literal_eval
import re
import h0rton.tdlmc_data
from h0rton.tdlmc_utils import tdlmc_metrics

__all__ = ['convert_to_dataframe', 'parse_closed_box', 'parse_open_box', 
'read_from_csv', 'format_results_for_tdlmc_metrics']

tdlmc_data_path = os.path.abspath(list(h0rton.tdlmc_data.__path__)[0])
"""str: directory path containing the TDLMC data

"""

def format_results_for_tdlmc_metrics(version_dir, out_dir, rung_id=2):
    """Format the BNN inference results so they can be read into the script that 
    generates the TDLMC metrics cornerplot

    Parameters
    ----------
    version_dir : str or os.path object
        path to the folder containing inference results
    rung_id : int
        TDLMC rung ID

    """
    label_to_id = {'A1 (0.5 HST orbit)': (4, ''), 
                   'A2 (1 HST orbit)': (3, ''), 
                   'A3 (2 HST orbits)': (2, ''), 
                   'B1 (89 doubles for 1 HST orbit)': (3, '_doubles'), 
                   'B2 (89 quads for 1 HST orbit)': (3, '_quads')}

    for label, (version_id, img) in label_to_id.items():
        summary = pd.read_csv(os.path.join(version_dir, 'summary.csv'), 
                              index_col=None)
        true_H0 = 70.0
        outside_rung = summary[ summary['id'] > (199)].index
        summary.drop(list(outside_rung), inplace=True)
        summary['keep'] = True # keep all lenses
        summary.loc[~summary['keep'], ['H0_mean', 'H0_std']] = -99
        summary['id'] = summary['id'].astype(int)
        
        if img != '':
            summary['is_quad'] = (summary['n_img'] == 4)
            n_test = np.min([len(summary[~summary['is_quad']]), 
                            len(summary[summary['is_quad']])])
            if img == '_doubles':
                summary = summary[~summary['is_quad']].iloc[:n_test]
            else:
                summary = summary[summary['is_quad']].iloc[:n_test]
                
        tdlmc_mean = summary['H0_mean'][summary['keep']]
        tdlmc_std = summary['H0_std'][summary['keep']]

        # Compute per-lens versions of the metrics
        summary['g'] = ((summary['H0_mean'] - true_H0)/summary['H0_std'])**2.0
        summary['log_g'] = np.log10(summary['g'])
        summary['p'] = (summary['H0_std']/true_H0)
        summary['a'] = (summary['H0_mean'] - true_H0)/true_H0

        # Test-set-side metrics
        G = tdlmc_metrics.get_goodness(tdlmc_mean,tdlmc_std, true_H0)
        P = tdlmc_metrics.get_precision(tdlmc_std, true_H0)
        A = tdlmc_metrics.get_accuracy(tdlmc_mean, true_H0)
        print("Goodness: ", G, "Log goodness: ", np.log10(G))
        print("Precision: ", P)
        print("Accuracy: ", A)
        print("Total combined", summary[summary['keep']].shape[0])
        print("Actually discarded", summary[~summary['keep']].shape[0])
        lens_name_formatting = lambda x: 'rung{:d}_seed{:d}'.format(rung_id, x)
        summary['rung_id'] = summary.id.apply(lens_name_formatting)
        summary = summary[['rung_id', 'H0_mean', 'H0_std']]
        summary.to_csv(os.path.join(out_dir, 'H0rton/{:s}.txt'.format(label)), 
                       header=None, index=None, sep=' ', mode='a')

def read_from_csv(csv_path):
    """Read a Pandas Dataframe from the combined csv file of TDLMC data while 
    evaluating all the relevant strings in each column as Python objects

    Parameters
    ----------
    csv_path : str
        path to the csv file generated using `convert_to_dataframe`

    Returns
    -------
    Pandas DataFrame
        the TDLMC data with correct Python objects

    """
    df = pd.read_csv(csv_path, index_col=False)
    # These are columns that are lists
    for list_col in [
                    'host_pos', 
                    'measured_td', 
                    'measured_td_err',
                    'agn_img_pos_x', 
                    'agn_img_pos_y', 
                    'agn_img_amp', 
                    'time_delays',
                    ]:
        df[list_col] = df[list_col].apply(literal_eval).apply(np.array)
    return df

def convert_to_dataframe(rung, save_csv_path):
    """Store the TDLMC closed and open boxes into a Pandas DataFrame and exports 
    to a csv file at the same location

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
            row['seed'] = seed
            row['seed_path'] = os.path.join(closed_code_dir, seed)
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

    # Manually add abcd_ordering_i
    df = df.sort_values('seed', axis=0)
    if rung == 1:
        df['abcd_ordering_i'] = np.array([[0, 1, 2, 3], #101
                                         [0, 1, 2, 3], #102
                                         [0, 1, 2, 3], #103
                                         [0, 1], #104
                                         [0, 1], #105
                                         [0, 1, 2, 3], #107
                                         [1, 0, 3, 2], #108
                                         [1, 2, 0, 3], #109
                                         [1, 2, 3, 0], #110
                                         [3, 1, 0, 2], #111
                                         [2, 0, 1, 3], #113
                                         [1, 0], #114
                                         [1, 3, 2, 0], #115
                                         [1, 0], #116
                                         [3, 2, 0, 1], #117
                                         [3, 1, 0, 2], #118
                                         ])
        df['H0'] = 74.151
    elif rung == 2:
        df['abcd_ordering_i'] = np.array([[0, 1, 2, 3], #119
                                         [0, 1, 2, 3], #120
                                         [0, 1, 2, 3], #121
                                         [0, 1, 2, 3], #122
                                         [0, 1, 2, 3], #123
                                         [0, 2, 1, 3], #124
                                         [0, 1], #125
                                         [0, 1], #126
                                         [3, 0, 1, 2], #127
                                         [3, 2, 0, 1], #128
                                         [3, 0, 1, 2], #129
                                         [2, 1, 0, 3], #130
                                         [3, 0, 2, 1], #131
                                         [1, 3, 2, 0], #132
                                         [1, 0], #133
                                         [0, 1], #134
                                         ])
        df['H0'] = 66.643
    else:
        raise NotImplementedError
    
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
    row_dict['measured_td'] = literal_eval(re.split(r'\(|\)', lines[7])[1])
    row_dict['measured_td_err'] = literal_eval(re.split(r'\(|\)', lines[7])[3])
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
    row_dict['H0'] = float(re.split(r':\s|km/s/Mpc', lines[3])[-2])
    row_dict['td_distance'] = float(re.split('ls:|Mpc', lines[5])[-2])
    row_dict['time_delays'] = literal_eval(re.split(r'\(|\)', lines[7])[1])
    row_dict['lens_mass'] = literal_eval(lines[11][7:])
    row_dict['ext_shear_e1e2'], row_dict['ext_shear_bphi'] = literal_eval(re.split(r'\(|\)', lines[12])[1])
    row_dict['lens_light'] = literal_eval(lines[14].split('\t')[1])
    row_dict['host_name'] = re.split(r'\(|\)|:|\t', lines[16])[2][1:]
    row_dict['host_pos'] = literal_eval(re.split(r'\(|\)|:|\t', lines[16])[-2])
    row_dict['host_mag'] = float(re.split(r'\t|\s', lines[17])[3])
    row_dict['host_r_eff'] = float(re.split(r'\t|\s', lines[17])[7])
    row_dict['agn_src_amp'] = float(lines[20].split()[-1])
    row_dict['agn_img_pos_x'] = literal_eval(re.split(r'\(|\)', lines[21])[1])
    row_dict['agn_img_pos_y'] = literal_eval(re.split(r'\(|\)', lines[21])[3])
    row_dict['agn_img_amp'] = literal_eval(re.split(r'\(|\)', lines[22])[1])
    row_dict['host_img_mag'] = re.split('plane: |mag|', lines[23])[3]
    row_dict['agn_img_mag'] = re.split('plane: |mag|', lines[23])[7]
    row_dict['vel_disp'] = float(re.split(r'km\/s| |\t', lines[25])[1])
    row_dict['kappa_ext'] = float(lines[27].split('\t')[1])

    return row_dict