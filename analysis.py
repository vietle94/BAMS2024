import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from cl61.func.study_case import process_raw

# %%
date = "20240529"
file_dir = f"/home/viet/Desktop/BAMS2024/data/raw/{date}/"
df_sample = process_raw(file_dir, "20240529 160000", "20240529 230000")
# df = df.sel(range=slice(0, 3000))


# %%

ref_mean = xr.open_dataset(
    "/media/viet/CL61/calibration/result/kenttarova/calibration_mean.nc"
)
df_mean_ref_sample = ref_mean.sel(
    internal_temperature_bins=df_sample.internal_temperature_bins, method='nearest'
).drop_vars("internal_temperature_bins")

# %%
df_sample["ppol_c"] = df_sample["ppol_r"] - df_mean_ref_sample["ppol_ref"]
df_sample["xpol_c"] = df_sample["xpol_r"] - df_mean_ref_sample["xpol_ref"]
df_sample["beta_c"] = (df_sample["ppol_c"] + df_sample["xpol_c"]) *  (df_sample.range**2)
# %%
df_sample["time"] = df_sample.time + pd.Timedelta(hours=3)
df_sample["range"] = df_sample.range + 273

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.pcolormesh(
    df_sample.time,
    df_sample.range,
    df_sample.p_pol.T,
    norm=LogNorm(vmin=1e-7, vmax=1e-5),
)
ax.grid()
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
ax.set_xlim(left = pd.to_datetime("2024-05-29T16:00"), right=pd.to_datetime("2024-05-29T23:00"))
ax.set_ylim(200, 1500)
ax.set_ylabel("Altitude (m)")
# fig.savefig(f"{date}_kenttarova_cl61d_beta_raw.png", dpi=600, bbox_inches="tight")

# %%
mytime = slice("2024-05-29T18:00", "2024-05-29T19:00")
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
ax[0].plot(df_sample.sel(time=mytime).p_pol.mean(dim="time"),
        df_sample.range)
ax[0].grid()
# ax[0].set_xscale("log")
ax[0].set_xlim(0, 1e-6)
ax[1].plot(df.sel(time=mytime).depolarisation.mean(dim="time"),
           df.range)
ax[1].grid()
ax[1].set_xlim(0, 0.2)
# ax[1].set_ylim(0, 2000)

# %%
mytime = slice("2024-05-29T19:00", "2024-05-29T20:00")
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
ax[0].plot(df_sample.sel(time=mytime).ppol_r.mean(dim="time"), df_sample.range)
ax[0].grid()
ax[1].plot(
    df_mean_ref_sample.sel(time=mytime).ppol_ref.mean(dim="time"),
    df_mean_ref_sample.range,
)
ax[1].grid()
ax[0].set_xlim(-2e-14, 2e-14)
ax[1].set_xlim(-2e-14, 2e-14)
# %%
