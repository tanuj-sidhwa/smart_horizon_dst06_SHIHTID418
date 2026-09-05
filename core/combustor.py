from .constants import CP,FUEL_LHV
def combustor_model(mass_flow_air,Tt3,Pt3,combustor_exit_temperature_K,combustor_pressure_drop_fraction,combustion_efficiency,air_fuel_ratio=None):
 fuel_flow=mass_flow_air/air_fuel_ratio if air_fuel_ratio and air_fuel_ratio>0 else mass_flow_air*CP*(combustor_exit_temperature_K-Tt3)/(combustion_efficiency*FUEL_LHV)
 return {"Tt4":combustor_exit_temperature_K,"Pt4":Pt3*(1-combustor_pressure_drop_fraction),"fuel_flow":fuel_flow}
