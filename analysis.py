import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from cl61.func.study_case import process_raw
from cl61.func.rayleigh import forward, molecular_backscatter
from cl61.func import rayleigh
import numpy as np
import glob

# %%
date = "20240531"
file_dir = f"/home/viet/Desktop/BAMS2024/data/raw/{date}/"
df_sample = process_raw(file_dir, "20240531 000000", "20240531 120000")
# df = df.sel(range=slice(0, 3000))

# %%
ref_mean = xr.open_dataset(
    "/media/viet/CL61/calibration/result/kenttarova/calibration_mean.nc"
)
df_mean_ref_sample = ref_mean.interp(
    internal_temperature_bins=df_sample.internal_temperature_bins,
    method="linear",
    kwargs={"fill_value": "extrapolate", "bounds_error": False},
).drop_vars("internal_temperature_bins")

# ref_mean.sel(
#     internal_temperature_bins=df_sample.internal_temperature_bins, method='nearest'
# ).drop_vars("internal_temperature_bins")

# %%
df_sample["ppol_c"] = df_sample["ppol_r"] - df_mean_ref_sample["ppol_ref"]
df_sample["xpol_c"] = df_sample["xpol_r"] - df_mean_ref_sample["xpol_ref"]
df_sample["beta_c"] = (df_sample["ppol_c"] + df_sample["xpol_c"]) * (df_sample.range**2)

df_sample["time"] = df_sample.time + pd.Timedelta(hours=3)
df_sample["range"] = df_sample.range + 276

# %%
fig, ax = plt.subplots(figsize=(9, 4))
p = ax.pcolormesh(
    df_sample.time,
    df_sample.range,
    df_sample.beta_c.T,
    norm=LogNorm(vmin=1e-7, vmax=1e-5),
)
fig.colorbar(p, ax=ax, label=r"$\beta'$ (m$^{-1}$ sr$^{-1}$)")
ax.grid()
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=3))
ax.set_ylim(200, 7000)
ax.set_ylabel("Altitude (m)")
# fig.savefig(f"{date}_kenttarova_cl61d_beta_raw.png", dpi=600, bbox_inches="tight")

# %%
time_slice = slice("2024-05-31T08:00", "2024-05-31T10:00")

# %%
model = xr.open_dataset(
    glob.glob(f"/home/viet/Desktop/BAMS2024/data/model/{date}*.nc")[0]
)
model = model.sel(time=time_slice).mean(dim="time")

model = model.swap_dims({"level": "height"})
model = model[["temperature", "pressure", "q"]]
model = model.drop_vars("level")
model = model.interp(height=df_sample.range)
mol_scatter = molecular_backscatter(
    2 * np.pi,
    model["temperature"],
    model["pressure"] / 100,  # Pa to hPa
)

beta_mol = mol_scatter / 1000
depo_mol = rayleigh.depo(
    rayleigh.f(910.55, 425, rayleigh.humidity_conversion(model["q"]))
)

beta_aerosol = forward(
    df_sample.beta_c.sel(time=time_slice).mean(dim="time"),
    beta_mol,
    50,
    1 / 1,
    df_sample.range,
)

depo_volume = df_sample["xpol_c"].sel(time=time_slice).mean(dim="time") / df_sample[
    "ppol_c"
].sel(time=time_slice).mean(dim="time")

beta_ratio = rayleigh.backscatter_ratio(beta_aerosol, beta_mol)

depo_aerosol = rayleigh.depo_aerosol(depo_volume, depo_mol, beta_ratio)

# %%
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4), sharex=True)
ax[0].plot(
    df_sample.sel(time=time_slice).beta_c.mean(dim="time"),
    df_sample.range,
    label=r"$\beta'$",
)
ax[0].plot(beta_mol, beta_mol.range, label=r"$\beta_{mol}$")
ax[0].set_xlim(-1e-6, 1e-6)
ax[1].plot(beta_aerosol, beta_aerosol.range, label=r"$\beta_{aerosol}$")
ax[0].set_ylabel("Altitude (m)")
for ax_ in ax:
    ax_.legend()
    ax_.grid()
# fig.savefig(f"figs/{date}_kenttarova_cl61d_beta_klett.png", dpi=600, bbox_inches="tight")

# %%
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
ax[0].plot(beta_aerosol, beta_aerosol.range, label=r"$\beta_{aerosol}$")

ax[1].plot(
    depo_volume,
    df_sample.range,
    ".",
    label="Depolarization ratio",
)
ax[1].set_xlim(0, 0.5)
ax[0].set_ylabel("Altitude (m)")
for ax_ in ax:
    ax_.grid()
    ax_.legend()

ax[1].set_ylim(0, 4000)
# fig.savefig(f"figs/{date}_kenttarova_cl61d_beta_depo.png", dpi=600, bbox_inches="tight")

# %%
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
ax[1].plot(
    depo_aerosol,
    df_sample.range,
    ".",
    label="Aerosol Depolarization ratio",
)

ax[1].plot(
    depo_volume,
    df_sample.range,
    ".",
    label="Depolarization ratio",
)
ax[1].set_xlim(0, 0.5)
ax[0].set_ylabel("Altitude (m)")
for ax_ in ax:
    ax_.grid()
    ax_.legend()

ax[1].set_ylim(0, 4000)
# %%
