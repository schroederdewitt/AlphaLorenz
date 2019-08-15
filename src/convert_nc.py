nc = netCDF4.Dataset("/vol/src/data/L96TwoLevel_ref.nc", mode='r')
for k in nc.variables.keys():
    nc[k][:].dump("/vol/src/data/{}.npy".format(k))
