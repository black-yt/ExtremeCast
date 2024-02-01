import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as colors



def pic_process(pred_d, pred_g, pred_e):
    weather_data_0 = pred_d[0,2].cpu().numpy() - 273.15
    weather_data_1 = pred_g[0,2].cpu().numpy() - 273.15
    weather_data_2 = pred_e[0,2].cpu().numpy() - 273.15

    fig = plt.figure(figsize=(120, 20))
    fig.suptitle("t2m visualization", fontsize=60)

    ax = fig.add_subplot(1, 3, 1, projection=ccrs.Robinson(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE, lw=2)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=50)
    ax.gridlines(linestyle='--')
    ax.set_global()
    ax.set_title("model_d", fontsize=60)
    im = ax.imshow(weather_data_0, cmap='coolwarm', extent=[0, 360, -90, 90], transform=ccrs.PlateCarree(), norm=colors.Normalize(-70, 50))
    cbar = plt.colorbar(
        mappable=im,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.8,
        cmap='coolwarm',
        extend="both")
    cbar.ax.tick_params(labelsize=50)
    
    
    ax = fig.add_subplot(1, 3, 2, projection=ccrs.Robinson(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE, lw=2)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=50)
    ax.gridlines(linestyle='--')
    ax.set_global()
    ax.set_title("model_g", fontsize=60)
    d = np.percentile(np.abs(weather_data_1-weather_data_0), 99.9)
    im = ax.imshow(weather_data_1-weather_data_0, cmap='RdBu_r', extent=[0, 360, -90, 90], transform=ccrs.PlateCarree(), norm=colors.Normalize(-d, d))
    cbar = plt.colorbar(
        mappable=im,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.8,
        cmap='RdBu_r',
        extend="both")
    cbar.ax.tick_params(labelsize=50)


    ax = fig.add_subplot(1, 3, 3, projection=ccrs.Robinson(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE, lw=2)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=50)
    ax.gridlines(linestyle='--')
    ax.set_global()
    ax.set_title("ExEnsemble", fontsize=60)
    d = np.percentile(np.abs(weather_data_2-weather_data_1), 99.9)
    im = ax.imshow(weather_data_2-weather_data_1, cmap='RdBu_r', extent=[0, 360, -90, 90], transform=ccrs.PlateCarree(), norm=colors.Normalize(-d, d))
    cbar = plt.colorbar(
        mappable=im,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.8,
        cmap='RdBu_r',
        extend="both")
    cbar.ax.tick_params(labelsize=50)


    plt.tight_layout()
    plt.savefig("./t2m_visualization.png")
    plt.close()

    print("Maximum and minimum temperatures in the modele_d prediction", round(weather_data_0.min(),2), round(weather_data_0.max(),2))
    print("Maximum and minimum temperatures in the modele_g prediction", round(weather_data_1.min(),2), round(weather_data_1.max(),2))
    print("Maximum and minimum temperatures in the ExEnsemble prediction", round(weather_data_2.min(),2), round(weather_data_2.max(),2))