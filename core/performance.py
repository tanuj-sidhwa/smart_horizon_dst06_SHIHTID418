from .constants import FUEL_LHV
def performance_model(mass_flow_air,fuel_flow,exhaust_velocity,exhaust_temperature,compressor_power):
 md=mass_flow_air+fuel_flow; thrust=md*exhaust_velocity; tsfc=fuel_flow/thrust if thrust>0 else float("inf"); eta=.5*md*exhaust_velocity**2/(fuel_flow*FUEL_LHV) if fuel_flow>0 else 0; return {"thrust_N":thrust,"fuel_flow_kg_s":fuel_flow,"tsfc_kg_N_s":tsfc,"exhaust_velocity_m_s":exhaust_velocity,"exhaust_temp_K":exhaust_temperature,"compressor_power_W":compressor_power,"thermal_efficiency":eta}
