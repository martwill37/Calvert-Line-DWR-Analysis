import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import matplotlib.dates as mdates
import cmocean as cm
import waypoint_distance as wd

def plot_mission(ds, file_pathway=None, lat_bounds=None, lon_bounds=None):
    """
    Plot a single glider path over bathymetry for the Calvert Line.

    Parameters:
    - ds: xarray.Dataset, the dataset to plot.
    - file_pathway: str, optional, used to extract mission name.
    - lon_bounds, lat_bounds: optional bounds for plotting.
    """
    topo_file = os.path.expanduser('~/Desktop/british_columbia_3_msl_2013.nc')

    # Extract lon/lat
    lons = ds['longitude'].values
    lats = ds['latitude'].values
    time_vals = ds['time'].values
    time_nums = mdates.date2num(time_vals)

    # Auto bounds if not provided
    if lon_bounds is None:
        lon_bounds = [lons.min() - 0.5, lons.max() + 0.5]
    if lat_bounds is None:
        lat_bounds = [lats.min() - 0.5, lats.max() + 0.5]

    # Load topo data
    topo = xr.open_dataset(topo_file)
    topo = topo.sel(
        lon=slice(lon_bounds[0], lon_bounds[1]),
        lat=slice(lat_bounds[0], lat_bounds[1]))

    # Set up plot
    fig, ax = plt.subplots(figsize=(1.8 * 6.4, 1.8 * 4.8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(lon_bounds + lat_bounds, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0)
    gl.top_labels = False
    gl.right_labels = False

    # Plot bathymetry
    topo_var = -topo['Band1']
    levels = np.linspace(0, 440, 45)
    contourf = ax.contourf(topo['lon'], topo['lat'], topo_var,
                           levels=levels, cmap=cm.cm.deep)
    fig.colorbar(contourf, ax=ax, label='Depth (m)')

    # Add 0 m elevation contour in black
    ax.contour(topo['lon'], topo['lat'], topo_var, levels=[0.5], colors='black', linewidths=1)

    # Plot glider track
    sc = ax.scatter(lons, lats, c=time_nums, cmap='seismic',
                    vmin=time_nums.min(), vmax=time_nums.max(),
                    s=5, transform=ccrs.PlateCarree(), label='Filtered Track')
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical')
    cbar.set_label('Date')
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.text(-128, 51.55, 'Calvert Island', fontsize=8, color='black',
        weight='bold', ha='center', va='center', rotation=0,
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    # Title
    if file_pathway:
        mission_name = os.path.basename(file_pathway).split('_')[0]  # Extract mission name
        ax.set_title(f'Cleaned Glider Mission Track – {mission_name}', fontsize=14)
    else:
        ax.set_title('Cleaned Glider Mission Track', fontsize=14)

topo_file=os.path.expanduser('~/Desktop/british_columbia_3_msl_2013.nc')
topo = xr.open_dataset(topo_file)

# Ensure topo is loaded globally for use in plot_section

def clean_mission(file_pathway):
    """
    Clean and plot a glider mission by separating outbound and return transects,
    removing redundant points, and plotting the results.
    """
    ds = xr.open_dataset(file_pathway)

    # Waypoints
    waypoint_lon = np.array([-127.950, -128.115, -128.243, -128.514, -128.646, -128.798])
    waypoint_lat = np.array([51.757, 51.705, 51.715, 51.450, 51.4165, 51.408])
    central_lat = 51.715

    # Distance projection
    alongx, acrossx, _ = wd.get_simple_distance(
        shiplon=ds['longitude'].values,
        shiplat=ds['latitude'].values,
        wplon=waypoint_lon,
        wplat=waypoint_lat,
        central_lat=central_lat)

    # Add along and across to ds
    ds = ds.assign(along=('time', alongx))
    ds = ds.assign(across=('time', acrossx))

    # Separate ds into out and return transects
    gradient_along = np.gradient(ds['along'])
    zero_indices = np.where(gradient_along == 0)[0]
    # Indices to figure out how the out and return should be separated
    # Requires gradient_along to have zeros in the middle, which are created from the projection function and it's bounds
    start = zero_indices[0]
    end = zero_indices[-1] + 1

    # Creating the masks
    out_mask = np.zeros_like(gradient_along, dtype=bool)
    return_mask = np.zeros_like(gradient_along, dtype=bool)
    out_mask[:start] = gradient_along[:start] > 0
    return_mask[end:] = gradient_along[end:] < 0

    # Defining new ds, one for each part of the trip
    ds_out = ds.sel(time=out_mask)
    ds_return = ds.sel(time=return_mask)

    # Iteratively remove points from glider loops (outbound)
    prev_len = -1
    while prev_len != len(ds_out['time']):
        prev_len = len(ds_out['time'])
        grad = np.gradient(ds_out['along'])
        keep_mask = grad > 0
        ds_out = ds_out.sel(time=keep_mask)

    # Iteratively remove points from glider loops (returnbound)
    prev_len = -1
    while prev_len != len(ds_return['time']):
        prev_len = len(ds_return['time'])
        grad = np.gradient(ds_return['along'])
        keep_mask = grad < 0
        ds_return = ds_return.sel(time=keep_mask)

    # Combine back into ds, plot. Data is now clean, not yet horizontally gridded
    ds = xr.concat([ds_out, ds_return], dim='time')
    # plot_mission(ds, file_pathway,lat_bounds = lat_bounds, lon_bounds = long_bounds)
    # plot_mission(ds_out, file_path = '/Users/martinwilliamson/Desktop/dfo-bb046-20210324_grid_delayed.nc',  lat_bounds = lat_bounds, long_bounds = long_bounds)
    # plot_mission(ds_return, file_path = '/Users/martinwilliamson/Desktop/dfo-bb046-20210324_grid_delayed.nc',  lat_bounds = lat_bounds, long_bounds = long_bounds)
    return ds, ds_out, ds_return

def interpolate(ds):
    """
    Interpolate the dataset to a regular grid along the 'along' dimension.
    """
    # Drop duplicates in the 'along' dimension
    _, index_unique = np.unique(ds['along'], return_index=True)
    ds = ds.isel(time=index_unique)
    
    # Ensure 'along' is a coordinate
    if 'along' not in ds.coords:
        raise ValueError("'along' must be a coordinate in the dataset.")

    # Create a new grid for interpolation
    along_grid = np.arange(ds['along'].min(), ds['along'].max(), 200)  # resolution of 200m

    # Save the original 'time' variable
    original_time = ds['time'].values

    # Swap dimensions and interpolate
    ds_swapped = ds.swap_dims({'time': 'along'})
    ds_interpolated = ds_swapped.interp(along=along_grid)

    # Reassign 'time' to the interpolated dataset
    # Use linear interpolation to estimate time values for the new 'along' grid
    interpolated_time = np.interp(along_grid, ds['along'].values, original_time.astype(float))
    ds_interpolated['time'] = ('along', interpolated_time.astype('datetime64[ns]'))

    return ds_interpolated

def plot_section(ds_out, ds_return, topo, file_pathway=None,  plot_interp = True, plot_time = False):
    """
    Plot temperature and potential density sections for both outbound and return transects.

    Parameters:
    - ds_out: xarray.Dataset, outbound transect dataset.
    - ds_return: xarray.Dataset, return transect dataset.
    - file_pathway: str, optional, used to extract mission name.
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 20,
        'figure.titlesize': 20})
    
    ds_out = ds_out.set_coords('along')
    ds_return = ds_return.set_coords('along')

    # Interpolate the outbound and return datasets
    ds_out = interpolate(ds_out)
    ds_return = interpolate(ds_return)

    # Extract mission name
    mission_name = os.path.basename(file_pathway).split('_')[0] if file_pathway else "Mission"

    # Define a consistent color range for temperature
    temp_min = 4  # Minimum temperature (°C)
    temp_max = 16  # Maximum temperature (°C)

    for ds, transect_type in zip([ds_out, ds_return], ["Outbound", "Return"]):
        # Extract variables
        along = ds['along'].values
        time = ds['time'].values
        depth = ds['depth'].values
        temperature = ds['temperature'].values
        pdens = ds['potential_density'].values - 1000  # Sigma-theta

        # Interpolate bathymetry
        interp_bathy = topo['Band1'].interp(
            lon=xr.DataArray(ds['longitude'].values, dims='time'),
            lat=xr.DataArray(ds['latitude'].values, dims='time'))
        bottom_depths = -interp_bathy.values  # 1D array

        if plot_interp:
            # ===== Plot Along-Transect Section =====
            fig, ax = plt.subplots(figsize=( 1.5 * 1.5 * 6.4, 1.5 * 4.8))
            cmap=cm.cm.thermal
            cf = ax.pcolormesh(along/1000, depth, temperature, shading='auto', cmap = cmap, vmin=temp_min, vmax=temp_max)
            ax.plot(along/1000, bottom_depths, color='black', linewidth=2, label='Bathymetry')

            # Isopycnal contours
            for levels, color, lw in [
                (np.linspace(24, 27, 7), 'black', 0.5),
                ([26.6], 'white', 2)]:
                cf_iso = ax.contour(along / 1000, depth, pdens, levels=levels, colors=color, linewidths=lw, linestyles='-')
                if lw != 0.3:
                    ax.clabel(cf_iso, fmt='%1.2f')

            ax.set_xlabel('Along-Transect Distance (km)')
            ax.set_ylabel('Depth (m)')
            ax.invert_yaxis()
            ax.set_title(f'Temperature Section ({transect_type} – {mission_name})')
            plt.colorbar(cf, ax=ax, label='Temperature (°C)')
            ax.legend(loc='lower right')
            plt.tight_layout()
            ax.set_ylim(410, 0)
            ax.set_xlim(0, 77)

        if plot_time:
            # ===== Plot Time Section =====
            fig, ax = plt.subplots(figsize=( 1.5 * 1.5 * 6.4, 1.5 * 4.8))
            cmap=cm.cm.thermal
            cf = ax.pcolormesh(time, depth, temperature, shading='auto', cmap=cmap, vmin=temp_min, vmax=temp_max)
            ax.plot(time, bottom_depths, color='black', linewidth=2, label='Bathymetry')

            # Isopycnal contours
            for levels, color, lw in [
                (np.linspace(24, 27, 7), 'black', 0.5),
                ([26.6], 'white', 2)]:
                cf_iso = ax.contour(time, depth, pdens, levels=levels, colors=color, linewidths=lw, linestyles='-')
                if lw != 0.3:
                    ax.clabel(cf_iso, fmt='%1.2f')

            ax.set_xlabel('Time')
            ax.set_ylabel('Depth (m)')
            ax.invert_yaxis()
            ax.set_title(f'Temperature Section (Time – {transect_type} – {mission_name})')
            plt.colorbar(cf, ax=ax, label='Temperature (°C)')
            ax.legend(loc='lower right')
            plt.tight_layout()
            ax.set_ylim(410, 0)

            # Format the time axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            # Invert x-axis for return transect in the time plot
            if transect_type == "Return":
                ax.invert_xaxis()

def process_glider_mission(file_pathway, lat_bounds=(51.2, 52), lon_bounds=(-128.9, -127.75), 
                           plot_time_section=True, plot_map=True):
    """
    Clean, plot on a map, and plot sections for both outbound and return glider trips.

    Args:
        file_pathway (str): Path to the glider netCDF file.
        lat_bounds (tuple): Latitude bounds for plotting.
        lon_bounds (tuple): Longitude bounds for plotting.
        plot_time_section (bool): Whether to plot the time section. Default is True.
        plot_map (bool): Whether to plot the map. Default is True.
    """
    # Clean and split the mission
    ds, ds_out, ds_return = clean_mission(file_pathway)

    # Plot the map if enabled
    if plot_map:
        plot_mission(ds, file_pathway, lat_bounds=lat_bounds, lon_bounds=lon_bounds)

    plot_section(ds_out, ds_return, topo, file_pathway=file_pathway, plot_interp = True, plot_time = False)

    # Plot sections for outbound and return (time) if enabled
    if plot_time_section:
        plot_section(ds_out, ds_return, topo, file_pathway=file_pathway, plot_interp = False, plot_time = True)

    return file_pathway

filepathway = (process_glider_mission)
