import pandas as pd

'''
Example script to save data to be read into example_fit_mcirsed.py.

'''


write_file = True

write_file_data = '../data/sample_sed_data.csv'

# define wavelengths, uncertainties, and redshifts for each galaxy. You may use whatever format you prefer. This is just an example 
d = {
    's24':[0.343, 0.727],
    's70':[7.700000, 17.799999],
    's100':[18.500000, 48.509998],
    's160':[47.240002, 101.091003],
    's250':[51.979401, 85.855797],
    's350':[41.887100, 67.430702],
    's500':[30.110800, 33.898998],
    's850':[4.000000, 6.0],
    's1100':[4.6, 2.5],
    'e24':[0.015, 0.036],
    'e70':[1.8, 2.4],
    'e100':[2.507, 1.758],
    'e160':[4.209, 5.838],
    'e250':[2.221000, 2.223000],
    'e350':[2.894, 2.932],
    'e500':[3.454, 4.075],
    'e850':[1.000000, 1.200000],
    'e1100':[2.0, 2.7],
    'z': [1.40620, 0.88550]
}
dF = pd.DataFrame(d)

if write_file is True:
    dF.to_csv(write_file_data) # write as a csv
